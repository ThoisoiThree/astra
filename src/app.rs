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
use glam::{Vec2, Vec3};
use molview::{
    DisplayColor, DisplayLevel, DisplayMode, DisplayState, DisplayStateData, ModeOverride,
    NamedSelectionStyle, RepresentationMask, VisibilityOverride,
    camera::{OrbitCamera, Viewport},
    command::{Command, Representation, parse_command},
    measurement::{MeasurementEndpoint, MeasurementLine},
    molecule::{MAX_DECOMPRESSED_STRUCTURE_SIZE, Molecule, MoleculeHierarchy, parse_structure},
    picking::AtomBvh,
    render::{PreparedCartoon, RenderError, Renderer, SurfaceIssue, prepare_cartoon},
    scene::{
        SCENE_EXTENSION, SCENE_FORMAT_NAME, SceneDocument, SceneHierarchyTarget,
        decode as decode_scene, encode as encode_scene, is_scene_document,
    },
    selection::{
        Selection, SelectionStatus, evaluate_with_named, rename_named_reference,
        resolve_named_expressions, validate_unique_name,
    },
};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalPosition},
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, ModifiersState},
    window::{Window, WindowAttributes, WindowId},
};

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
    CameraUpdate, FocusRequest, HierarchySelectionGesture, InspectionTarget, ManagerAction,
    PivotRequest, SessionTab, UiActions, UiInfo, UiState,
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

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(runtime) = &mut self.runtime {
            runtime.poll_background_jobs();
            runtime.autosave_due_documents();
            if let Some(deadline) = runtime.next_autosave_deadline() {
                event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
            } else {
                event_loop.set_control_flow(ControlFlow::Wait);
            }
        }
    }
}

struct Runtime {
    window: Arc<Window>,
    renderer: Renderer,
    egui_context: egui::Context,
    egui_state: egui_winit::State,
    molecule: Option<Molecule>,
    hierarchy: Option<MoleculeHierarchy>,
    atom_bvh: Option<AtomBvh>,
    display: Option<DisplayState>,
    named_selections: BTreeMap<String, Selection>,
    named_selection_expressions: BTreeMap<String, String>,
    named_selection_styles: BTreeMap<String, NamedSelectionStyle>,
    named_selection_statuses: BTreeMap<String, SelectionStatus>,
    measurement_lines: Vec<MeasurementLine>,
    next_measurement_id: u64,
    hierarchy_names: BTreeMap<InspectionTarget, String>,
    inspection: Option<InspectionTarget>,
    hierarchy_selection: BTreeSet<InspectionTarget>,
    hierarchy_selection_anchor: Option<InspectionTarget>,
    focus_description: String,
    pivot_description: String,
    loaded_filename: Option<String>,
    molecule_id: Option<String>,
    scene_path: Option<PathBuf>,
    dirty: bool,
    recovery_path: Option<PathBuf>,
    autosave_due: Option<Instant>,
    document_version: u64,
    needs_cartoon_refresh: bool,
    next_pick_request_id: u64,
    pending_pick: Option<PendingPick>,
    inactive_sessions: Vec<DocumentSession>,
    session_order: Vec<u64>,
    active_session_id: u64,
    next_session_id: u64,
    fetch_receiver: Option<Receiver<FetchEvent>>,
    fetching_pdb_id: Option<String>,
    fetch_progress: FetchProgress,
    fetch_cancel: Option<Arc<AtomicBool>>,
    job_sender: SyncSender<JobRequest>,
    job_receiver: Receiver<JobEvent>,
    background_jobs: BTreeMap<u64, BackgroundJob>,
    next_job_id: u64,
    camera: OrbitCamera,
    ui: UiState,
    cursor: Option<PhysicalPosition<f64>>,
    viewport: Viewport,
    left_drag: bool,
    left_drag_distance: f32,
    right_drag: bool,
    modifiers: ModifiersState,
    focused: bool,
    occluded: bool,
    undo_history: VecDeque<EditOperation>,
    redo_history: VecDeque<EditOperation>,
}

struct DocumentSession {
    id: u64,
    molecule: Option<Molecule>,
    hierarchy: Option<MoleculeHierarchy>,
    atom_bvh: Option<AtomBvh>,
    display: Option<DisplayState>,
    named_selections: BTreeMap<String, Selection>,
    named_selection_expressions: BTreeMap<String, String>,
    named_selection_styles: BTreeMap<String, NamedSelectionStyle>,
    named_selection_statuses: BTreeMap<String, SelectionStatus>,
    measurement_lines: Vec<MeasurementLine>,
    next_measurement_id: u64,
    hierarchy_names: BTreeMap<InspectionTarget, String>,
    inspection: Option<InspectionTarget>,
    hierarchy_selection: BTreeSet<InspectionTarget>,
    hierarchy_selection_anchor: Option<InspectionTarget>,
    focus_description: String,
    pivot_description: String,
    loaded_filename: Option<String>,
    molecule_id: Option<String>,
    scene_path: Option<PathBuf>,
    dirty: bool,
    recovery_path: Option<PathBuf>,
    autosave_due: Option<Instant>,
    document_version: u64,
    needs_cartoon_refresh: bool,
    camera: OrbitCamera,
    undo_history: VecDeque<EditOperation>,
    redo_history: VecDeque<EditOperation>,
}

impl DocumentSession {
    fn empty(id: u64, viewport: Viewport) -> Self {
        let aspect = viewport.width / viewport.height.max(1.0);
        Self {
            id,
            molecule: None,
            hierarchy: None,
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
            needs_cartoon_refresh: false,
            camera: OrbitCamera::new(aspect),
            undo_history: VecDeque::new(),
            redo_history: VecDeque::new(),
        }
    }

    fn tab(&self) -> SessionTab {
        SessionTab {
            id: self.id,
            label: session_label(self.molecule_id.as_deref(), self.loaded_filename.as_deref()),
            dirty: self.dirty,
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
            .context("recovery session has no structure")?;
        let display = self
            .display
            .as_ref()
            .context("recovery session has no display state")?;
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
}

#[derive(Debug, Clone, Copy, Default)]
struct FetchProgress {
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
    bytes_per_second: f64,
}

#[derive(Debug, Clone, Copy)]
struct PendingPick {
    request_id: u64,
    session_id: u64,
    document_version: u64,
    cpu_fallback: Option<usize>,
}

struct BackgroundJob {
    session_id: u64,
    version: u64,
    kind: JobKind,
    stage: String,
    progress: f32,
    cancel: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobKind {
    Load,
    Save,
    Cartoon,
}

enum JobRequest {
    Load {
        id: u64,
        session_id: u64,
        version: u64,
        path: PathBuf,
        cancel: Arc<AtomicBool>,
    },
    Save {
        id: u64,
        session_id: u64,
        version: u64,
        path: PathBuf,
        document: Box<SceneDocument>,
        cancel: Arc<AtomicBool>,
    },
    Cartoon {
        id: u64,
        session_id: u64,
        version: u64,
        molecule: Box<Molecule>,
        display: Box<DisplayState>,
        cancel: Arc<AtomicBool>,
    },
}

enum JobEvent {
    Progress {
        id: u64,
        stage: &'static str,
        progress: f32,
    },
    Complete {
        id: u64,
        result: Box<std::result::Result<JobOutput, String>>,
    },
}

enum JobOutput {
    Loaded(Box<LoadedPayload>),
    Saved(PathBuf),
    Cartoon(PreparedCartoon),
    Cancelled,
}

enum LoadedPayload {
    Structure {
        filename: String,
        molecule_id: String,
        molecule: Molecule,
        hierarchy: MoleculeHierarchy,
        atom_bvh: AtomBvh,
        display: Box<DisplayState>,
    },
    Scene {
        filename: String,
        path: PathBuf,
        document: Box<SceneDocument>,
        hierarchy: MoleculeHierarchy,
        atom_bvh: AtomBvh,
    },
}

#[derive(Debug)]
enum FetchEvent {
    Progress(FetchProgress),
    Finished(Result<PathBuf, String>),
}

#[derive(Debug, Clone, PartialEq)]
struct EditTransaction {
    display: Option<DisplayStateData>,
    named_selections: BTreeMap<String, NamedSelectionRecord>,
    measurements: BTreeMap<u64, IndexedMeasurement>,
    hierarchy_names: BTreeMap<InspectionTarget, String>,
    workspace: WorkspaceSelection,
}

#[derive(Debug, Clone, PartialEq)]
struct NamedSelectionRecord {
    selection: Selection,
    expression: Option<String>,
    style: Option<NamedSelectionStyle>,
}

#[derive(Debug, Clone, PartialEq)]
struct IndexedMeasurement {
    index: usize,
    line: MeasurementLine,
}

#[derive(Debug, Clone, PartialEq)]
struct WorkspaceSelection {
    inspection: Option<InspectionTarget>,
    hierarchy_selection: BTreeSet<InspectionTarget>,
    hierarchy_selection_anchor: Option<InspectionTarget>,
}

#[derive(Debug, Clone, PartialEq)]
struct ValueChange<T> {
    before: T,
    after: T,
}

#[derive(Debug, Clone, PartialEq)]
struct NamedSelectionChange {
    name: String,
    value: ValueChange<Option<NamedSelectionRecord>>,
}

#[derive(Debug, Clone, PartialEq)]
struct MeasurementChange {
    id: u64,
    value: ValueChange<Option<IndexedMeasurement>>,
}

#[derive(Debug, Clone, PartialEq)]
struct HierarchyNameChange {
    target: InspectionTarget,
    value: ValueChange<Option<String>>,
}

#[derive(Debug, Clone, PartialEq)]
enum EditChange {
    Display(Box<ValueChange<Option<DisplayStateData>>>),
    NamedSelection(Box<NamedSelectionChange>),
    Measurement(Box<MeasurementChange>),
    HierarchyName(Box<HierarchyNameChange>),
    Workspace(Box<ValueChange<WorkspaceSelection>>),
}

#[derive(Debug, Clone, PartialEq, Default)]
struct EditOperation {
    changes: Vec<EditChange>,
}

#[derive(Debug, Clone, Copy)]
enum HistoryDirection {
    Undo,
    Redo,
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
        };
        if let Some(path) = initial_path
            && let Err(error) = runtime.start_load_path(&path)
        {
            runtime.ui.latest_error = Some(error.to_string());
        }
        runtime.offer_recovery_files();
        runtime.window.request_redraw();
        Ok(runtime)
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        self.poll_background_jobs();
        self.poll_fetch_result();
        self.poll_pick_result();
        let egui_response = self.egui_state.on_window_event(&self.window, &event);
        if egui_response.repaint {
            self.window.request_redraw();
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
                            self.mark_dirty();
                            self.window.request_redraw();
                        } else if self.left_drag && self.left_drag_distance > 4.0 {
                            self.camera.orbit(delta);
                            self.mark_dirty();
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
                self.mark_dirty();
                self.window.request_redraw();
            }
            WindowEvent::RedrawRequested
                if self.focused
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
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let session_tabs = self.session_tabs();
        let background_job = self
            .background_jobs
            .iter()
            .find(|(_, job)| job.session_id == self.active_session_id);
        let info = UiInfo {
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
            self.display.as_ref(),
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
            self.mark_dirty();
        }
        if actions.reset_colors
            && let (Some(molecule), Some(display)) = (&self.molecule, &mut self.display)
        {
            let before = EditableSnapshot {
                display: Some(display.clone()),
                named_selections: self.named_selections.clone(),
                named_selection_expressions: self.named_selection_expressions.clone(),
                named_selection_styles: self.named_selection_styles.clone(),
                measurement_lines: self.measurement_lines.clone(),
                hierarchy_names: self.hierarchy_names.clone(),
                inspection: self.inspection,
                hierarchy_selection: self.hierarchy_selection.clone(),
                hierarchy_selection_anchor: self.hierarchy_selection_anchor,
            };
            display.reset_colors(molecule);
            for style in self.named_selection_styles.values_mut() {
                style.color = None;
            }
            display.replace_named_layers(&named_display_layers(
                &self.named_selections,
                &self.named_selection_styles,
            ));
            self.needs_cartoon_refresh = true;
            self.ui.latest_error = None;
            self.commit_edit(before);
        }
        if let Some(command) = actions.execute {
            match parse_command(&command).map_err(anyhow::Error::from) {
                Ok(command) => {
                    let before = self.editable_snapshot();
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
            let before = self.editable_snapshot();
            self.handle_manager_action(action);
            self.commit_edit(before);
        }
        if let Some(update) = actions.camera_update {
            self.apply_camera_update(update);
            self.mark_dirty();
        }
        if let Some(request) = actions.focus_request {
            match self.resolve_focus(request) {
                Ok((point, description)) => {
                    self.camera.depth_of_field.focus_point = point;
                    self.focus_description = description;
                    self.mark_dirty();
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
                    self.mark_dirty();
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

    fn request_close_session(&mut self, id: u64) {
        if id != self.active_session_id {
            self.activate_session(id);
        }
        if id == self.active_session_id && self.confirm_active_document_close() {
            self.close_session(id);
        }
    }

    fn request_exit(&mut self) -> bool {
        let ids = self.session_order.clone();
        for id in ids {
            if id != self.active_session_id {
                self.activate_session(id);
            }
            if self.molecule.is_some() && !self.confirm_active_document_close() {
                return false;
            }
        }
        true
    }

    fn confirm_active_document_close(&mut self) -> bool {
        if !self.dirty {
            return true;
        }
        let label = session_label(self.molecule_id.as_deref(), self.loaded_filename.as_deref());
        let answer = rfd::MessageDialog::new()
            .set_parent(self.window.as_ref())
            .set_level(rfd::MessageLevel::Warning)
            .set_title("Unsaved Molecule scene")
            .set_description(format!("Save changes to “{label}”?"))
            .set_buttons(rfd::MessageButtons::YesNoCancelCustom(
                "Save".into(),
                "Discard".into(),
                "Cancel".into(),
            ))
            .show();
        match answer {
            rfd::MessageDialogResult::Custom(value) if value == "Save" => {
                let result = if let Some(path) = self.scene_path.clone() {
                    self.save_scene_to(path)
                } else {
                    self.save_scene_as()
                };
                if let Err(error) = result {
                    self.ui.latest_error = Some(error.to_string());
                    return false;
                }
                !self.dirty
            }
            rfd::MessageDialogResult::Yes => {
                let result = if let Some(path) = self.scene_path.clone() {
                    self.save_scene_to(path)
                } else {
                    self.save_scene_as()
                };
                if let Err(error) = result {
                    self.ui.latest_error = Some(error.to_string());
                    return false;
                }
                !self.dirty
            }
            rfd::MessageDialogResult::Custom(value) if value == "Discard" => {
                self.remove_recovery_file();
                true
            }
            rfd::MessageDialogResult::No => {
                self.remove_recovery_file();
                true
            }
            _ => false,
        }
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
        let filename = path.file_name().map_or_else(
            || path.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        );
        self.begin_new_session();
        self.molecule = None;
        self.hierarchy = None;
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
                kind: JobKind::Load,
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
                            self.apply_loaded_job(job.session_id, job.version, *payload);
                        }
                        Ok(JobOutput::Saved(path)) => {
                            self.apply_saved_job(job.session_id, job.version, path);
                        }
                        Ok(JobOutput::Cartoon(prepared)) => {
                            if job.session_id == self.active_session_id
                                && job.version == self.document_version
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
    }

    fn schedule_cartoon_job(&mut self) {
        let (Some(molecule), Some(display)) = (&self.molecule, &self.display) else {
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
        let cancel = Arc::new(AtomicBool::new(false));
        let request = JobRequest::Cartoon {
            id,
            session_id: self.active_session_id,
            version: self.document_version,
            molecule: Box::new(molecule.clone()),
            display: Box::new(display.clone()),
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
                version: self.document_version,
                kind: JobKind::Cartoon,
                stage: "Queued ribbon geometry".into(),
                progress: 0.0,
                cancel,
            },
        );
    }

    fn apply_loaded_job(&mut self, session_id: u64, version: u64, payload: LoadedPayload) {
        let original_session = self.active_session_id;
        if session_id != original_session {
            self.activate_session(session_id);
        }
        if self.active_session_id == session_id && self.document_version == version {
            match payload {
                LoadedPayload::Structure {
                    filename,
                    molecule_id,
                    molecule,
                    hierarchy,
                    atom_bvh,
                    display,
                } => {
                    self.loaded_filename = Some(filename);
                    self.molecule_id = Some(molecule_id);
                    self.scene_path = None;
                    self.molecule = Some(molecule);
                    self.hierarchy = Some(hierarchy);
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
                    self.loaded_filename = Some(filename);
                    self.scene_path = Some(path);
                    self.molecule = Some(molecule);
                    self.hierarchy = Some(hierarchy);
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
            self.dirty = false;
            self.recovery_path = None;
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

    fn load_scene_document(
        &mut self,
        document: SceneDocument,
        filename: String,
        scene_path: Option<PathBuf>,
        dirty: bool,
        recovery_path: Option<PathBuf>,
    ) -> Result<()> {
        self.begin_new_session();
        self.ui.document_changed();
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
            ..
        } = document;
        let molecule_id = if source_name.trim().is_empty() {
            molecule_id_from_filename(&filename)
        } else {
            molecule_id_from_filename(&source_name)
        };
        let hierarchy = MoleculeHierarchy::from_molecule(&molecule);
        let atom_bvh = AtomBvh::build(&molecule);
        camera.set_viewport(self.viewport);
        self.renderer.update_measurements(&measurement_lines);
        self.next_measurement_id = measurement_lines
            .iter()
            .map(|line| line.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        self.loaded_filename = Some(filename);
        self.molecule_id = Some(molecule_id);
        self.scene_path = scene_path;
        self.molecule = Some(molecule);
        self.hierarchy = Some(hierarchy);
        self.atom_bvh = Some(atom_bvh);
        self.display = Some(display);
        self.named_selections = named_selections;
        self.named_selection_expressions = named_selection_expressions;
        self.named_selection_styles = named_selection_styles;
        self.recalculate_named_selections();
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
        self.hierarchy_selection_anchor = hierarchy_selection_anchor.map(inspection_target);
        self.focus_description = focus_description;
        self.pivot_description = pivot_description;
        self.camera = camera;
        self.undo_history.clear();
        self.redo_history.clear();
        self.dirty = dirty;
        self.recovery_path = recovery_path;
        self.autosave_due = None;
        self.needs_cartoon_refresh = true;
        self.schedule_cartoon_job();
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
            return;
        }
        push_history(&mut self.undo_history, operation);
        self.redo_history.clear();
        self.mark_dirty();
    }

    fn undo(&mut self) -> bool {
        let Some(operation) = self.undo_history.pop_back() else {
            return false;
        };
        self.apply_edit_operation(&operation, HistoryDirection::Undo);
        push_history(&mut self.redo_history, operation);
        self.mark_dirty();
        true
    }

    fn redo(&mut self) -> bool {
        let Some(operation) = self.redo_history.pop_back() else {
            return false;
        };
        self.apply_edit_operation(&operation, HistoryDirection::Redo);
        push_history(&mut self.undo_history, operation);
        self.mark_dirty();
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

    fn offer_recovery_files(&mut self) {
        let Ok(directory) = recovery_directory() else {
            return;
        };
        let Ok(entries) = fs::read_dir(directory) else {
            return;
        };
        let mut paths: Vec<_> = entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with(".recovery.mol"))
            })
            .collect();
        paths.sort();
        for path in paths {
            let contents = match read_local_file_limited(&path) {
                Ok(contents) => contents,
                Err(error) => {
                    self.ui.latest_error = Some(format!(
                        "could not inspect recovery {}: {error:#}",
                        path.display()
                    ));
                    continue;
                }
            };
            let document = match decode_scene(&contents) {
                Ok(document) => document,
                Err(error) => {
                    self.ui.latest_error = Some(format!(
                        "invalid recovery document {}: {error}",
                        path.display()
                    ));
                    continue;
                }
            };
            let label = if document.source_name.trim().is_empty() {
                "untitled molecule".to_owned()
            } else {
                document.source_name.clone()
            };
            let answer = rfd::MessageDialog::new()
                .set_parent(self.window.as_ref())
                .set_level(rfd::MessageLevel::Warning)
                .set_title("Recovered Molecule scene")
                .set_description(format!(
                    "An autosaved scene for “{label}” was found. Restore it?"
                ))
                .set_buttons(rfd::MessageButtons::YesNoCancelCustom(
                    "Restore".into(),
                    "Discard".into(),
                    "Later".into(),
                ))
                .show();
            match answer {
                rfd::MessageDialogResult::Custom(value) if value == "Restore" => {
                    let filename = format!("{label} (Recovered)");
                    if let Err(error) =
                        self.load_scene_document(document, filename, None, true, Some(path.clone()))
                    {
                        self.ui.latest_error = Some(error.to_string());
                    }
                }
                rfd::MessageDialogResult::Yes => {
                    let filename = format!("{label} (Recovered)");
                    if let Err(error) =
                        self.load_scene_document(document, filename, None, true, Some(path.clone()))
                    {
                        self.ui.latest_error = Some(error.to_string());
                    }
                }
                rfd::MessageDialogResult::Custom(value) if value == "Discard" => {
                    let _ = fs::remove_file(path);
                }
                rfd::MessageDialogResult::No => {
                    let _ = fs::remove_file(path);
                }
                _ => {}
            }
        }
    }

    fn apply_command(&mut self, command: Command) -> Result<()> {
        if self.molecule.is_none() || self.display.is_none() {
            anyhow::bail!("load a PDB file before executing commands");
        }
        match command {
            Command::Select {
                name,
                source,
                selection,
            } => {
                let selection = evaluate_with_named(
                    &selection,
                    self.molecule.as_ref().unwrap_or_else(|| unreachable!()),
                    &self.named_selections,
                )?;
                if let Some(name) = name {
                    validate_unique_name(&self.named_selections, &name, None)
                        .map_err(anyhow::Error::msg)?;
                    self.named_selections
                        .insert(name.clone(), selection.clone());
                    self.named_selection_expressions
                        .insert(name.clone(), source);
                    self.named_selection_styles.entry(name.clone()).or_default();
                    self.recalculate_named_selections();
                }
                if let Some(display) = &mut self.display {
                    display.set_selection(selection.flags().clone());
                }
                self.inspection = None;
                self.hierarchy_selection.clear();
                self.hierarchy_selection_anchor = None;
            }
            Command::Color { color, selection } => {
                let indices: Vec<_> = evaluate_with_named(
                    &selection,
                    self.molecule.as_ref().unwrap_or_else(|| unreachable!()),
                    &self.named_selections,
                )?
                .indices()
                .collect();
                if let Some(display) = &mut self.display {
                    display.set_color_override(&indices, DisplayLevel::Atom, Some(color.0));
                }
            }
            Command::Show {
                representation,
                selection,
            } => {
                let mask = representation_mask(representation);
                let selected = evaluate_with_named(
                    &selection,
                    self.molecule.as_ref().unwrap_or_else(|| unreachable!()),
                    &self.named_selections,
                )?;
                if let Some(display) = &mut self.display {
                    display.set_representation(selected.indices(), mask, true);
                }
            }
            Command::Hide {
                representation,
                selection,
            } => {
                let mask = representation_mask(representation);
                let selected = evaluate_with_named(
                    &selection,
                    self.molecule.as_ref().unwrap_or_else(|| unreachable!()),
                    &self.named_selections,
                )?;
                if let Some(display) = &mut self.display {
                    display.set_representation(selected.indices(), mask, false);
                }
            }
        }
        self.refresh_instances();
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

    fn request_gpu_pick(&mut self, position: PhysicalPosition<f64>) {
        let point = Vec2::new(position.x as f32, position.y as f32);
        if !self.viewport.contains(point) || self.molecule.is_none() {
            return;
        }
        let cpu_fallback = self
            .camera
            .screen_ray(point, self.viewport)
            .and_then(|ray| {
                self.molecule.as_ref().and_then(|molecule| {
                    let display = self.display.as_ref();
                    self.atom_bvh.as_ref()?.pick_with_radius(
                        molecule,
                        ray,
                        |index| {
                            display.is_none_or(|display| {
                                let Some(atom) = molecule.atoms.get(index) else {
                                    return false;
                                };
                                display.visible.get(index).unwrap_or(false)
                                    && (matches!(
                                        display.modes.get(index),
                                        Some(DisplayMode::BallAndStick | DisplayMode::Toon)
                                    ) || atom.hetero
                                        || matches!(atom.name.as_str(), "CA" | "P"))
                            })
                        },
                        |index, atom| {
                            if display.and_then(|display| display.modes.get(index))
                                == Some(&DisplayMode::Toon)
                            {
                                atom.element.van_der_waals_radius()
                            } else {
                                (atom.element.van_der_waals_radius() * 0.32).max(0.32)
                            }
                        },
                    )
                })
            });
        let request_id = self.next_pick_request_id;
        self.next_pick_request_id = self.next_pick_request_id.wrapping_add(1);
        self.renderer.request_pick(
            request_id,
            position.x.max(0.0) as u32,
            position.y.max(0.0) as u32,
        );
        self.pending_pick = Some(PendingPick {
            request_id,
            session_id: self.active_session_id,
            document_version: self.document_version,
            cpu_fallback,
        });
    }

    fn poll_pick_result(&mut self) {
        let Some((request_id, result)) = self.renderer.poll_pick() else {
            return;
        };
        let Some(pending) = self.pending_pick.take() else {
            return;
        };
        if pending.request_id != request_id
            || pending.session_id != self.active_session_id
            || pending.document_version != self.document_version
        {
            return;
        }
        let picked = result.unwrap_or(pending.cpu_fallback);
        if picked.is_some_and(|index| {
            self.molecule
                .as_ref()
                .is_none_or(|molecule| index >= molecule.atoms.len())
        }) {
            return;
        }
        let before = self.editable_snapshot();
        match picked {
            Some(atom_index) => {
                let target = InspectionTarget::Atom(atom_index);
                self.hierarchy_selection.clear();
                self.hierarchy_selection.insert(target);
                self.hierarchy_selection_anchor = Some(target);
                self.select_indices(vec![atom_index], Some(target));
            }
            None => {
                self.hierarchy_selection.clear();
                self.hierarchy_selection_anchor = None;
                self.select_indices(Vec::new(), None);
            }
        }
        self.commit_edit(before);
        self.window.request_redraw();
    }

    fn handle_manager_action(&mut self, action: ManagerAction) {
        match action {
            ManagerAction::SelectHierarchy { target, gesture } => {
                self.select_hierarchy(target, gesture);
            }
            ManagerAction::SetColor { targets, color } => {
                let operations: Vec<_> = targets
                    .into_iter()
                    .map(|target| self.display_target(target))
                    .collect();
                if let Some(display) = &mut self.display {
                    for (indices, level) in operations {
                        display.set_color_override(&indices, level, color);
                    }
                }
                self.refresh_instances();
            }
            ManagerAction::SetVisibility { targets, state } => {
                let operations: Vec<_> = targets
                    .into_iter()
                    .map(|target| self.display_target(target))
                    .map(|(indices, level)| (indices, level, state))
                    .collect();
                if let Some(display) = &mut self.display {
                    display.set_visibility_overrides(&operations);
                }
                self.refresh_instances();
            }
            ManagerAction::SetMode { targets, state } => {
                let operations: Vec<_> = targets
                    .into_iter()
                    .map(|target| self.display_target(target))
                    .map(|(indices, level)| (indices, level, state))
                    .collect();
                if let Some(display) = &mut self.display {
                    display.set_mode_overrides(&operations);
                }
                self.refresh_instances();
            }
            ManagerAction::PropagateColor(targets) => self.propagate_color(&targets),
            ManagerAction::PropagateVisibility(targets) => self.propagate_visibility(&targets),
            ManagerAction::PropagateMode(targets) => self.propagate_mode(&targets),
            ManagerAction::SetNamedColor { name, color } => {
                if self.named_selections.contains_key(&name) {
                    self.named_selection_styles.entry(name).or_default().color = color;
                    self.rebuild_named_display_layers();
                }
            }
            ManagerAction::SetNamedVisibility { name, state } => {
                if self.named_selections.contains_key(&name) {
                    self.named_selection_styles
                        .entry(name)
                        .or_default()
                        .visibility = state;
                    self.rebuild_named_display_layers();
                }
            }
            ManagerAction::SetNamedMode { name, state } => {
                if self.named_selections.contains_key(&name) {
                    let global = self
                        .display
                        .as_ref()
                        .map_or(DisplayMode::Cartoon, |display| display.global_mode);
                    self.named_selection_styles.entry(name).or_default().mode =
                        if state.mode() == Some(global) {
                            ModeOverride::Inherit
                        } else {
                            state
                        };
                    self.rebuild_named_display_layers();
                }
            }
            ManagerAction::PropagateNamedColor { name, color } => {
                let indices = self.named_selection_indices(&name);
                if let Some(display) = &mut self.display {
                    display.set_color_overrides_forced(&[(indices, DisplayLevel::Atom, color)]);
                }
                self.refresh_instances();
            }
            ManagerAction::PropagateNamedVisibility { name, state } => {
                let indices = self.named_selection_indices(&name);
                if let Some(display) = &mut self.display {
                    display.set_visibility_override(&indices, DisplayLevel::Atom, state);
                }
                self.refresh_instances();
            }
            ManagerAction::PropagateNamedMode { name, state } => {
                let indices = self.named_selection_indices(&name);
                if let Some(display) = &mut self.display {
                    display.set_mode_override(&indices, DisplayLevel::Atom, state);
                }
                self.refresh_instances();
            }
            ManagerAction::SelectNamedSubset {
                indices,
                inspection,
            } => {
                self.hierarchy_selection.clear();
                self.hierarchy_selection_anchor = None;
                self.select_indices(indices, Some(inspection));
            }
            ManagerAction::SetColoringMode(mode) => {
                if let (Some(molecule), Some(display)) = (&self.molecule, &mut self.display) {
                    display.set_coloring_mode(molecule, mode);
                }
                self.refresh_instances();
            }
            ManagerAction::SetGlobalMode(mode) => {
                if let Some(display) = &mut self.display {
                    display.set_global_mode(mode);
                }
                let redundant = ModeOverride::from_mode(mode);
                for style in self.named_selection_styles.values_mut() {
                    if style.mode == redundant {
                        style.mode = ModeOverride::Inherit;
                    }
                }
                self.rebuild_named_display_layers();
            }
            ManagerAction::SetAmbientOcclusion(settings) => {
                if let Some(display) = &mut self.display {
                    display.set_ambient_occlusion(settings);
                }
            }
            ManagerAction::SetUniformColor(color) => {
                if let (Some(molecule), Some(display)) = (&self.molecule, &mut self.display) {
                    display.set_uniform_color(molecule, color);
                }
                self.refresh_instances();
            }
            ManagerAction::ActivateNamed(name) => {
                if let Some(selection) = self.named_selections.get(&name) {
                    self.hierarchy_selection.clear();
                    self.hierarchy_selection_anchor = None;
                    let flags = selection.flags().clone();
                    if let Some(display) = &mut self.display {
                        display.set_selection(flags);
                    }
                    self.inspection = None;
                    self.refresh_instances();
                }
            }
            ManagerAction::UpdateNamedExpression { name, expression } => {
                let result = (|| -> Result<()> {
                    self.molecule
                        .as_ref()
                        .context("load a PDB file before editing a selection")?;
                    if !self.named_selections.contains_key(&name) {
                        anyhow::bail!("named selection '{name}' does not exist");
                    }
                    self.named_selection_expressions
                        .insert(name.clone(), expression);
                    self.named_selection_styles.entry(name).or_default();
                    self.recalculate_named_selections();
                    self.hierarchy_selection.clear();
                    self.hierarchy_selection_anchor = None;
                    self.inspection = None;
                    Ok(())
                })();
                match result {
                    Ok(()) => self.ui.latest_error = None,
                    Err(error) => self.ui.latest_error = Some(error.to_string()),
                }
            }
            ManagerAction::RemoveNamed(name) => {
                self.named_selections.remove(&name);
                self.named_selection_expressions.remove(&name);
                self.named_selection_styles.remove(&name);
                self.named_selection_statuses.remove(&name);
                self.recalculate_named_selections();
            }
            ManagerAction::CreateMeasurement {
                first_selection,
                second_selection,
            } => match self.create_measurement(&first_selection, &second_selection) {
                Ok(()) => self.ui.latest_error = None,
                Err(error) => self.ui.latest_error = Some(error.to_string()),
            },
            ManagerAction::SetMeasurementColor { id, color } => {
                if let Some(line) = self.measurement_lines.iter_mut().find(|line| line.id == id) {
                    line.color = color;
                    self.renderer.update_measurements(&self.measurement_lines);
                }
            }
            ManagerAction::SetMeasurementVisibility { id, state } => {
                if let Some(line) = self.measurement_lines.iter_mut().find(|line| line.id == id) {
                    line.visibility = state;
                    self.renderer.update_measurements(&self.measurement_lines);
                }
            }
            ManagerAction::SetMeasurementLabelSize { id, size } => {
                if let Some(line) = self.measurement_lines.iter_mut().find(|line| line.id == id) {
                    line.set_label_size(size);
                }
            }
            ManagerAction::SetMeasurementThickness { id, thickness } => {
                if let Some(line) = self.measurement_lines.iter_mut().find(|line| line.id == id) {
                    line.set_thickness(thickness);
                    self.renderer.update_measurements(&self.measurement_lines);
                }
            }
            ManagerAction::RemoveMeasurement(id) => {
                self.measurement_lines.retain(|line| line.id != id);
                self.renderer.update_measurements(&self.measurement_lines);
            }
            ManagerAction::SaveCurrentSelection(name) => {
                let result = (|| -> Result<()> {
                    validate_unique_name(&self.named_selections, &name, None)
                        .map_err(anyhow::Error::msg)?;
                    let flags = self
                        .display
                        .as_ref()
                        .context("load a structure before saving a selection")?
                        .selection
                        .clone();
                    if !flags.iter().any(|selected| selected) {
                        anyhow::bail!("select at least one atom first");
                    }
                    self.named_selections
                        .insert(name.clone(), Selection::from_mask(flags));
                    self.named_selection_styles.entry(name).or_default();
                    self.recalculate_named_selections();
                    Ok(())
                })();
                match result {
                    Ok(()) => self.ui.latest_error = None,
                    Err(error) => self.ui.latest_error = Some(error.to_string()),
                }
            }
            ManagerAction::RenameHierarchy { target, name } => {
                if let Some(name) = name {
                    self.hierarchy_names.insert(target, name);
                } else {
                    self.hierarchy_names.remove(&target);
                }
            }
            ManagerAction::RenameNamedSelection { old_name, new_name } => {
                let result = (|| -> Result<()> {
                    if old_name == new_name {
                        return Ok(());
                    }
                    validate_unique_name(&self.named_selections, &new_name, Some(&old_name))
                        .map_err(anyhow::Error::msg)?;
                    let selection = self
                        .named_selections
                        .remove(&old_name)
                        .with_context(|| format!("named selection '{old_name}' does not exist"))?;
                    self.named_selections.insert(new_name.clone(), selection);
                    if let Some(expression) = self.named_selection_expressions.remove(&old_name) {
                        self.named_selection_expressions
                            .insert(new_name.clone(), expression);
                    }
                    if let Some(style) = self.named_selection_styles.remove(&old_name) {
                        self.named_selection_styles.insert(new_name.clone(), style);
                    }
                    for expression in self.named_selection_expressions.values_mut() {
                        if let Ok(renamed) =
                            rename_named_reference(expression, &old_name, &new_name)
                        {
                            *expression = renamed;
                        }
                    }
                    self.named_selection_statuses.remove(&old_name);
                    self.recalculate_named_selections();
                    Ok(())
                })();
                match result {
                    Ok(()) => self.ui.latest_error = None,
                    Err(error) => self.ui.latest_error = Some(error.to_string()),
                }
            }
            ManagerAction::RenameMeasurement { id, name } => {
                if let Some(line) = self.measurement_lines.iter_mut().find(|line| line.id == id) {
                    line.name = name;
                }
            }
        }
    }

    fn create_measurement(&mut self, first_name: &str, second_name: &str) -> Result<()> {
        let molecule = self
            .molecule
            .as_ref()
            .context("load a structure before creating a distance line")?;
        let first = measurement_endpoint(
            molecule,
            first_name,
            self.named_selections
                .get(first_name)
                .with_context(|| format!("named selection '{first_name}' does not exist"))?,
        )?;
        let second = measurement_endpoint(
            molecule,
            second_name,
            self.named_selections
                .get(second_name)
                .with_context(|| format!("named selection '{second_name}' does not exist"))?,
        )?;
        if first.position.distance(second.position) <= f32::EPSILON {
            anyhow::bail!("the two measurement endpoints occupy the same point");
        }
        self.measurement_lines.push(MeasurementLine::new(
            self.next_measurement_id,
            first,
            second,
        ));
        self.next_measurement_id += 1;
        self.renderer.update_measurements(&self.measurement_lines);
        Ok(())
    }

    fn select_hierarchy(&mut self, target: InspectionTarget, gesture: HierarchySelectionGesture) {
        let range = self
            .hierarchy_selection_anchor
            .and_then(|anchor| self.hierarchy_range(anchor, target));
        let inspection = update_hierarchy_selection(
            &mut self.hierarchy_selection,
            &mut self.hierarchy_selection_anchor,
            target,
            gesture,
            range.as_deref(),
        );
        let targets: Vec<_> = self.hierarchy_selection.iter().copied().collect();
        let indices = targets
            .into_iter()
            .flat_map(|target| self.display_target(target).0)
            .collect();
        self.select_indices(indices, inspection);
    }

    fn named_selection_indices(&self, name: &str) -> Vec<usize> {
        self.named_selections
            .get(name)
            .map(|selection| selection.indices().collect())
            .unwrap_or_default()
    }

    fn rebuild_named_display_layers(&mut self) {
        let layers = named_display_layers(&self.named_selections, &self.named_selection_styles);
        if let Some(display) = &mut self.display {
            display.replace_named_layers(&layers);
        }
        self.refresh_instances();
    }

    fn recalculate_named_selections(&mut self) {
        let Some(molecule) = &self.molecule else {
            self.named_selection_statuses.clear();
            return;
        };
        let resolution = resolve_named_expressions(
            molecule,
            &self.named_selection_expressions,
            &self.named_selections,
        );
        self.named_selections = resolution.selections;
        self.named_selection_statuses = resolution.statuses;
        self.rebuild_named_display_layers();
    }

    fn hierarchy_range(
        &self,
        anchor: InspectionTarget,
        target: InspectionTarget,
    ) -> Option<Vec<InspectionTarget>> {
        if std::mem::discriminant(&anchor) != std::mem::discriminant(&target) {
            return None;
        }
        let hierarchy = self.hierarchy.as_ref()?;
        let ordered: Vec<InspectionTarget> = match target {
            InspectionTarget::Chain(_) => (0..hierarchy.chains.len())
                .map(InspectionTarget::Chain)
                .collect(),
            InspectionTarget::Residue { .. } => hierarchy
                .chains
                .iter()
                .enumerate()
                .flat_map(|(chain_index, chain)| {
                    (0..chain.residues.len()).map(move |residue_index| InspectionTarget::Residue {
                        chain_index,
                        residue_index,
                    })
                })
                .collect(),
            InspectionTarget::Atom(_) => hierarchy
                .chains
                .iter()
                .flat_map(|chain| &chain.residues)
                .flat_map(|residue| residue.atom_indices.iter().copied())
                .map(InspectionTarget::Atom)
                .collect(),
        };
        inclusive_target_range(&ordered, anchor, target)
    }

    fn propagate_color(&mut self, targets: &[InspectionTarget]) {
        let mut operations = Vec::<(Vec<usize>, DisplayLevel, DisplayColor)>::new();
        for &target in targets {
            let (source_indices, source_level) = self.display_target(target);
            let Some(source_atom) = source_indices.first().copied() else {
                continue;
            };
            let Some(color) = self
                .display
                .as_ref()
                .map(|display| display.color_at_level(source_atom, source_level))
            else {
                continue;
            };
            for child in self.descendant_targets(target) {
                let (indices, level) = self.display_target(child);
                operations.push((indices, level, color));
            }
        }
        if let Some(display) = &mut self.display {
            display.set_color_overrides_forced(&operations);
        }
        self.refresh_instances();
    }

    fn propagate_visibility(&mut self, targets: &[InspectionTarget]) {
        let mut operations = Vec::<(Vec<usize>, DisplayLevel, VisibilityOverride)>::new();
        for &target in targets {
            let (source_indices, source_level) = self.display_target(target);
            let Some(source_atom) = source_indices.first().copied() else {
                continue;
            };
            let Some(display) = &self.display else {
                continue;
            };
            let direct = display.visibility_override(source_atom, source_level);
            let state = match direct {
                VisibilityOverride::Inherit => {
                    if display.visible.get(source_atom).unwrap_or(true) {
                        VisibilityOverride::Show
                    } else {
                        VisibilityOverride::Hide
                    }
                }
                state => state,
            };
            for child in self.descendant_targets(target) {
                let (indices, level) = self.display_target(child);
                operations.push((indices, level, state));
            }
        }
        if let Some(display) = &mut self.display {
            display.set_visibility_overrides(&operations);
        }
        self.refresh_instances();
    }

    fn propagate_mode(&mut self, targets: &[InspectionTarget]) {
        let mut operations = Vec::<(Vec<usize>, DisplayLevel, ModeOverride)>::new();
        for &target in targets {
            let (source_indices, source_level) = self.display_target(target);
            let Some(source_atom) = source_indices.first().copied() else {
                continue;
            };
            let Some(mode) = self
                .display
                .as_ref()
                .map(|display| display.mode_at_level(source_atom, source_level))
            else {
                continue;
            };
            for child in self.descendant_targets(target) {
                let (indices, level) = self.display_target(child);
                operations.push((indices, level, ModeOverride::from_mode(mode)));
            }
        }
        if let Some(display) = &mut self.display {
            display.set_mode_overrides(&operations);
        }
        self.refresh_instances();
    }

    fn descendant_targets(&self, target: InspectionTarget) -> Vec<InspectionTarget> {
        match target {
            InspectionTarget::Chain(chain_index) => self
                .hierarchy
                .as_ref()
                .and_then(|hierarchy| hierarchy.chains.get(chain_index))
                .map(|chain| {
                    chain
                        .residues
                        .iter()
                        .enumerate()
                        .flat_map(|(residue_index, residue)| {
                            std::iter::once(InspectionTarget::Residue {
                                chain_index,
                                residue_index,
                            })
                            .chain(
                                residue
                                    .atom_indices
                                    .iter()
                                    .copied()
                                    .map(InspectionTarget::Atom),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default(),
            InspectionTarget::Residue {
                chain_index,
                residue_index,
            } => self
                .hierarchy
                .as_ref()
                .and_then(|hierarchy| hierarchy.residue(chain_index, residue_index))
                .map(|residue| {
                    residue
                        .atom_indices
                        .iter()
                        .copied()
                        .map(InspectionTarget::Atom)
                        .collect()
                })
                .unwrap_or_default(),
            InspectionTarget::Atom(_) => Vec::new(),
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
        let Some(display) = &mut self.display else {
            return;
        };
        display.set_selection_from_bools(flags);
        self.inspection = inspection;
        self.refresh_instances();
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
        self.needs_cartoon_refresh = true;
        self.schedule_cartoon_job();
        self.renderer.update_measurements(&self.measurement_lines);
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

fn molecule_id_from_structure(contents: &[u8], filename: &str) -> String {
    if let Ok(text) = std::str::from_utf8(contents) {
        for line in text.lines().take(200) {
            if line.starts_with("HEADER")
                && let Some(id) = line.get(62..66)
                && is_structure_id(id.trim())
            {
                return id.trim().to_ascii_uppercase();
            }
            if let Some(value) = line.trim().strip_prefix("_entry.id")
                && let Some(id) = value.split_whitespace().next()
            {
                let id = id.trim_matches(['\'', '"']);
                if is_structure_id(id) {
                    return id.to_ascii_uppercase();
                }
            }
        }
    }
    molecule_id_from_filename(filename)
}

fn molecule_id_from_filename(filename: &str) -> String {
    let mut name = Path::new(filename)
        .file_name()
        .map_or(filename, |name| name.to_str().unwrap_or(filename));
    if name
        .rsplit_once('.')
        .is_some_and(|(_, extension)| extension.eq_ignore_ascii_case("gz"))
    {
        name = name.rsplit_once('.').map_or(name, |(stem, _)| stem);
    }
    let stem = name.rsplit_once('.').map_or(name, |(stem, _)| stem).trim();
    if stem.is_empty() {
        "Untitled".into()
    } else if is_structure_id(stem) {
        stem.to_ascii_uppercase()
    } else {
        stem.to_owned()
    }
}

fn is_structure_id(value: &str) -> bool {
    let value = value.trim();
    (value.len() == 4 && value.bytes().all(|byte| byte.is_ascii_alphanumeric()))
        || (value.len() == 12
            && value.to_ascii_uppercase().starts_with("PDB_")
            && value[4..].bytes().all(|byte| byte.is_ascii_alphanumeric()))
}

fn session_label(molecule_id: Option<&str>, filename: Option<&str>) -> String {
    molecule_id
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .or_else(|| filename.map(molecule_id_from_filename))
        .unwrap_or_else(|| "Untitled".into())
}

fn normalize_pdb_id(value: &str) -> Result<String> {
    let id = value.trim().to_ascii_uppercase();
    let legacy = id.len() == 4 && id.bytes().all(|byte| byte.is_ascii_alphanumeric());
    let extended = id.len() == 12
        && id.starts_with("PDB_")
        && id[4..].bytes().all(|byte| byte.is_ascii_alphanumeric());
    if legacy || extended {
        Ok(id)
    } else {
        anyhow::bail!(
            "'{value}' is not a valid PDB ID; use four letters/digits such as 4R8P or an extended pdb_00004hhb ID"
        )
    }
}

fn pdb_download_directory() -> Result<PathBuf> {
    let home = env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .context("could not determine the home directory")?;
    Ok(PathBuf::from(home).join("downloads").join("pdb"))
}

#[cfg(test)]
fn download_pdb(id: &str, directory: &Path) -> Result<PathBuf> {
    download_pdb_with_progress(id, directory, &AtomicBool::new(false), |_| {})
}

fn download_pdb_with_progress(
    id: &str,
    directory: &Path,
    cancel: &AtomicBool,
    mut report_progress: impl FnMut(FetchProgress),
) -> Result<PathBuf> {
    let filename = format!("{id}.cif");
    let url = pdb_download_url(id);
    let agent = pdb_agent();
    let (supports_ranges, advertised_size) = download_metadata(&agent, &url);
    if advertised_size.is_some_and(|length| length > MAX_FETCH_SIZE) {
        anyhow::bail!("PDB entry {id} exceeds the 512 MiB download limit");
    }
    let started = Instant::now();
    report_progress(FetchProgress {
        downloaded_bytes: 0,
        total_bytes: advertised_size,
        bytes_per_second: 0.0,
    });
    let contents = if supports_ranges
        && advertised_size.is_some_and(|length| length >= PARALLEL_FETCH_MIN_SIZE)
    {
        match download_pdb_ranges(
            id,
            &url,
            advertised_size.unwrap_or_default(),
            cancel,
            started,
            &mut report_progress,
        ) {
            Ok(contents) => contents,
            Err(error) if cancel.load(Ordering::Relaxed) => return Err(error),
            Err(_) => {
                report_progress(FetchProgress {
                    downloaded_bytes: 0,
                    total_bytes: advertised_size,
                    bytes_per_second: 0.0,
                });
                download_pdb_stream(
                    id,
                    &url,
                    &agent,
                    advertised_size,
                    cancel,
                    started,
                    &mut report_progress,
                )?
            }
        }
    } else {
        download_pdb_stream(
            id,
            &url,
            &agent,
            advertised_size,
            cancel,
            started,
            &mut report_progress,
        )?
    };
    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    report_progress(FetchProgress {
        downloaded_bytes: contents.len() as u64,
        total_bytes: Some(contents.len() as u64),
        bytes_per_second: contents.len() as f64 / elapsed,
    });
    parse_structure(&contents, &filename)
        .with_context(|| format!("RCSB returned invalid structure data for {id}"))?;
    if cancel.load(Ordering::Relaxed) {
        anyhow::bail!(FETCH_CANCELLED);
    }
    fs::create_dir_all(directory)
        .with_context(|| format!("could not create {}", directory.display()))?;
    let path = directory.join(filename);
    atomic_write(&path, &contents)
        .with_context(|| format!("could not save downloaded structure to {}", path.display()))?;
    Ok(path)
}

fn pdb_agent() -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(60)))
        .https_only(true)
        .build();
    ureq::Agent::new_with_config(config)
}

fn download_metadata(agent: &ureq::Agent, url: &str) -> (bool, Option<u64>) {
    let Ok(response) = agent
        .head(url)
        .header("User-Agent", concat!("molview/", env!("CARGO_PKG_VERSION")))
        .call()
    else {
        return (false, None);
    };
    let supports_ranges = response
        .headers()
        .get("accept-ranges")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case("bytes"));
    let size = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok());
    (supports_ranges, size)
}

fn download_pdb_stream(
    id: &str,
    url: &str,
    agent: &ureq::Agent,
    advertised_size: Option<u64>,
    cancel: &AtomicBool,
    started: Instant,
    report_progress: &mut impl FnMut(FetchProgress),
) -> Result<Vec<u8>> {
    let mut response = agent
        .get(url)
        .header("User-Agent", concat!("molview/", env!("CARGO_PKG_VERSION")))
        .call()
        .with_context(|| format!("could not fetch PDB entry {id} from RCSB"))?;
    let total_bytes = response.body().content_length().or(advertised_size);
    if total_bytes.is_some_and(|length| length > MAX_FETCH_SIZE) {
        anyhow::bail!("PDB entry {id} exceeds the 512 MiB download limit");
    }
    let capacity = total_bytes
        .unwrap_or_default()
        .min(MAX_FETCH_SIZE)
        .try_into()
        .unwrap_or(0);
    let mut contents = Vec::with_capacity(capacity);
    let mut reader = response.body_mut().as_reader();
    let mut buffer = [0_u8; FETCH_BUFFER_SIZE];
    let mut last_report = started;
    loop {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!(FETCH_CANCELLED);
        }
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("could not download PDB entry {id}"))?;
        if read == 0 {
            break;
        }
        if contents.len().saturating_add(read) as u64 > MAX_FETCH_SIZE {
            anyhow::bail!("PDB entry {id} exceeds the 512 MiB download limit");
        }
        contents.extend_from_slice(&buffer[..read]);
        let now = Instant::now();
        if now.duration_since(last_report) >= Duration::from_millis(100) {
            let elapsed = now.duration_since(started).as_secs_f64().max(0.001);
            report_progress(FetchProgress {
                downloaded_bytes: contents.len() as u64,
                total_bytes,
                bytes_per_second: contents.len() as f64 / elapsed,
            });
            last_report = now;
        }
    }
    Ok(contents)
}

enum RangeDownloadEvent {
    Downloaded(usize),
    Finished {
        index: usize,
        contents: std::result::Result<Vec<u8>, String>,
    },
}

fn download_pdb_ranges(
    id: &str,
    url: &str,
    total_bytes: u64,
    cancel: &AtomicBool,
    started: Instant,
    report_progress: &mut impl FnMut(FetchProgress),
) -> Result<Vec<u8>> {
    let worker_count = PARALLEL_FETCH_WORKERS.min(total_bytes.max(1) as usize);
    let range_size = total_bytes.div_ceil(worker_count as u64);
    let (sender, receiver) = mpsc::channel();
    let stop = AtomicBool::new(false);

    thread::scope(|scope| -> Result<Vec<u8>> {
        for index in 0..worker_count {
            let start = index as u64 * range_size;
            let end = (start + range_size - 1).min(total_bytes - 1);
            let sender = sender.clone();
            let stop = &stop;
            scope.spawn(move || {
                let result = download_pdb_range(id, url, start, end, cancel, stop, &sender)
                    .map_err(|error| error.to_string());
                if result.is_err() {
                    stop.store(true, Ordering::Relaxed);
                }
                let _ = sender.send(RangeDownloadEvent::Finished {
                    index,
                    contents: result,
                });
            });
        }
        drop(sender);

        let mut parts = vec![None; worker_count];
        let mut completed = 0;
        let mut downloaded_bytes = 0_u64;
        let mut last_report = started;
        let mut first_error = None;
        while completed < worker_count {
            if cancel.load(Ordering::Relaxed) {
                stop.store(true, Ordering::Relaxed);
                anyhow::bail!(FETCH_CANCELLED);
            }
            match receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(RangeDownloadEvent::Downloaded(bytes)) => {
                    downloaded_bytes = downloaded_bytes.saturating_add(bytes as u64);
                }
                Ok(RangeDownloadEvent::Finished { index, contents }) => {
                    completed += 1;
                    match contents {
                        Ok(contents) => parts[index] = Some(contents),
                        Err(_) if cancel.load(Ordering::Relaxed) => {
                            anyhow::bail!(FETCH_CANCELLED);
                        }
                        Err(error) if error == FETCH_CANCELLED => {}
                        Err(error) if first_error.is_none() => first_error = Some(error),
                        Err(_) => {}
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    anyhow::bail!("parallel PDB download stopped unexpectedly");
                }
            }
            let now = Instant::now();
            if now.duration_since(last_report) >= Duration::from_millis(100)
                || completed == worker_count
            {
                let elapsed = now.duration_since(started).as_secs_f64().max(0.001);
                report_progress(FetchProgress {
                    downloaded_bytes,
                    total_bytes: Some(total_bytes),
                    bytes_per_second: downloaded_bytes as f64 / elapsed,
                });
                last_report = now;
            }
        }

        if let Some(error) = first_error {
            anyhow::bail!("could not download byte range for PDB entry {id}: {error}");
        }
        let capacity = usize::try_from(total_bytes).unwrap_or_default();
        let mut contents = Vec::with_capacity(capacity);
        for part in parts {
            let part = part.context("parallel PDB download returned an incomplete file")?;
            contents.extend_from_slice(&part);
        }
        if contents.len() as u64 != total_bytes {
            anyhow::bail!(
                "parallel PDB download returned {} bytes instead of {total_bytes}",
                contents.len()
            );
        }
        Ok(contents)
    })
}

fn download_pdb_range(
    id: &str,
    url: &str,
    start: u64,
    end: u64,
    cancel: &AtomicBool,
    stop: &AtomicBool,
    sender: &Sender<RangeDownloadEvent>,
) -> Result<Vec<u8>> {
    let mut response = pdb_agent()
        .get(url)
        .header("User-Agent", concat!("molview/", env!("CARGO_PKG_VERSION")))
        .header("Range", format!("bytes={start}-{end}"))
        .call()
        .with_context(|| format!("could not fetch PDB entry {id} from RCSB"))?;
    if response.status().as_u16() != 206 {
        anyhow::bail!(
            "RCSB did not honor the requested byte range ({})",
            response.status()
        );
    }
    let expected = end - start + 1;
    let capacity = usize::try_from(expected).unwrap_or_default();
    let mut contents = Vec::with_capacity(capacity);
    let mut reader = response.body_mut().as_reader();
    let mut buffer = [0_u8; FETCH_BUFFER_SIZE];
    loop {
        if cancel.load(Ordering::Relaxed) || stop.load(Ordering::Relaxed) {
            anyhow::bail!(FETCH_CANCELLED);
        }
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("could not download PDB entry {id}"))?;
        if read == 0 {
            break;
        }
        if contents.len().saturating_add(read) as u64 > expected {
            anyhow::bail!("RCSB returned too many bytes for a requested range");
        }
        contents.extend_from_slice(&buffer[..read]);
        let _ = sender.send(RangeDownloadEvent::Downloaded(read));
    }
    if contents.len() as u64 != expected {
        anyhow::bail!(
            "RCSB returned {} bytes for a {expected}-byte range",
            contents.len()
        );
    }
    Ok(contents)
}

fn pdb_download_url(id: &str) -> String {
    format!("https://files.rcsb.org/download/{id}.cif")
}

fn molecule_path(mut path: PathBuf) -> PathBuf {
    let has_extension = path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case(SCENE_EXTENSION));
    if !has_extension {
        path.set_extension(SCENE_EXTENSION);
    }
    path
}

fn scene_target(target: InspectionTarget) -> SceneHierarchyTarget {
    match target {
        InspectionTarget::Chain(index) => SceneHierarchyTarget::Chain(index),
        InspectionTarget::Residue {
            chain_index,
            residue_index,
        } => SceneHierarchyTarget::Residue {
            chain_index,
            residue_index,
        },
        InspectionTarget::Atom(index) => SceneHierarchyTarget::Atom(index),
    }
}

fn inspection_target(target: SceneHierarchyTarget) -> InspectionTarget {
    match target {
        SceneHierarchyTarget::Chain(index) => InspectionTarget::Chain(index),
        SceneHierarchyTarget::Residue {
            chain_index,
            residue_index,
        } => InspectionTarget::Residue {
            chain_index,
            residue_index,
        },
        SceneHierarchyTarget::Atom(index) => InspectionTarget::Atom(index),
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

fn measurement_endpoint(
    molecule: &Molecule,
    selection_name: &str,
    selection: &Selection,
) -> Result<MeasurementEndpoint> {
    let indices: Vec<_> = selection.indices().collect();
    let first_index = *indices
        .first()
        .with_context(|| format!("named selection '{selection_name}' is empty"))?;
    let first = molecule
        .atoms
        .get(first_index)
        .with_context(|| format!("named selection '{selection_name}' refers to a missing atom"))?;
    if indices.len() == 1 {
        return Ok(MeasurementEndpoint {
            position: first.position,
            description: format!("{selection_name}: atom #{} {}", first.serial, first.name),
        });
    }

    let one_residue = indices.iter().all(|index| {
        molecule.atoms.get(*index).is_some_and(|atom| {
            atom.chain_id == first.chain_id
                && atom.residue_name == first.residue_name
                && atom.residue_number == first.residue_number
                && atom.insertion_code == first.insertion_code
        })
    });
    if !one_residue {
        anyhow::bail!(
            "named selection '{selection_name}' must contain one atom or atoms from exactly one residue/base"
        );
    }
    Ok(MeasurementEndpoint {
        position: centroid(molecule, &indices).with_context(|| {
            format!("named selection '{selection_name}' contains no valid atoms")
        })?,
        description: format!(
            "{selection_name}: {} {} / chain {}",
            first.residue_name, first.residue_number, first.chain_id
        ),
    })
}

fn representation_mask(representation: Representation) -> RepresentationMask {
    match representation {
        Representation::Spheres => RepresentationMask::SPHERES,
        Representation::Sticks => RepresentationMask::STICKS,
    }
}

fn push_history(history: &mut VecDeque<EditableSnapshot>, snapshot: EditableSnapshot) {
    if history.len() == EDIT_HISTORY_LIMIT {
        history.pop_front();
    }
    history.push_back(snapshot);
}

fn named_display_layers(
    selections: &BTreeMap<String, Selection>,
    styles: &BTreeMap<String, NamedSelectionStyle>,
) -> Vec<(Vec<usize>, NamedSelectionStyle)> {
    selections
        .iter()
        .map(|(name, selection)| {
            (
                selection.indices().collect(),
                styles.get(name).copied().unwrap_or_default(),
            )
        })
        .collect()
}

fn update_hierarchy_selection(
    selection: &mut BTreeSet<InspectionTarget>,
    anchor: &mut Option<InspectionTarget>,
    target: InspectionTarget,
    gesture: HierarchySelectionGesture,
    range: Option<&[InspectionTarget]>,
) -> Option<InspectionTarget> {
    match gesture {
        HierarchySelectionGesture::Replace => {
            selection.clear();
            selection.insert(target);
            *anchor = Some(target);
        }
        HierarchySelectionGesture::Range => {
            selection.clear();
            if let Some(range) = range {
                selection.extend(range.iter().copied());
            } else {
                selection.insert(target);
                *anchor = Some(target);
            }
        }
        HierarchySelectionGesture::Toggle => {
            if !selection.insert(target) {
                selection.remove(&target);
            }
            if anchor.is_none() {
                *anchor = Some(target);
            }
        }
    }
    if selection.contains(&target) {
        Some(target)
    } else {
        selection.iter().next_back().copied()
    }
}

fn inclusive_target_range(
    ordered: &[InspectionTarget],
    anchor: InspectionTarget,
    target: InspectionTarget,
) -> Option<Vec<InspectionTarget>> {
    let anchor_index = ordered.iter().position(|candidate| *candidate == anchor)?;
    let target_index = ordered.iter().position(|candidate| *candidate == target)?;
    let start = anchor_index.min(target_index);
    let end = anchor_index.max(target_index);
    Some(ordered[start..=end].to_vec())
}

fn read_local_file_limited(path: &Path) -> Result<Vec<u8>> {
    let file =
        fs::File::open(path).with_context(|| format!("could not open {}", path.display()))?;
    let size = file
        .metadata()
        .with_context(|| format!("could not inspect {}", path.display()))?
        .len();
    if size > MAX_LOCAL_FILE_SIZE {
        anyhow::bail!(
            "{} is {} bytes; the local-file safety limit is {} bytes",
            path.display(),
            size,
            MAX_LOCAL_FILE_SIZE
        );
    }
    let mut contents = Vec::with_capacity(size.min(8 * 1024 * 1024) as usize);
    file.take(MAX_LOCAL_FILE_SIZE.saturating_add(1))
        .read_to_end(&mut contents)
        .with_context(|| format!("could not read {}", path.display()))?;
    if contents.len() as u64 > MAX_LOCAL_FILE_SIZE {
        anyhow::bail!(
            "{} grew beyond the {} byte local-file safety limit while it was being read",
            path.display(),
            MAX_LOCAL_FILE_SIZE
        );
    }
    Ok(contents)
}

fn background_worker(
    requests: Receiver<JobRequest>,
    events: Sender<JobEvent>,
    window: Arc<Window>,
) {
    while let Ok(request) = requests.recv() {
        let (id, result) = match request {
            JobRequest::Load {
                id,
                session_id,
                version,
                path,
                cancel,
            } => (
                id,
                load_in_background(id, session_id, version, &path, &cancel, &events, &window)
                    .map(|payload| JobOutput::Loaded(Box::new(payload))),
            ),
            JobRequest::Save {
                id,
                session_id,
                version,
                path,
                document,
                cancel,
            } => {
                let _ = (session_id, version);
                send_job_progress(&events, &window, id, "Encoding Molecule scene", 0.25);
                let result = if cancel.load(Ordering::Relaxed) {
                    Ok(JobOutput::Cancelled)
                } else {
                    encode_scene(&document)
                        .context("could not encode scene")
                        .and_then(|contents| {
                            if cancel.load(Ordering::Relaxed) {
                                return Ok(JobOutput::Cancelled);
                            }
                            send_job_progress(
                                &events,
                                &window,
                                id,
                                "Synchronizing scene file",
                                0.8,
                            );
                            atomic_write(&path, &contents)?;
                            Ok(JobOutput::Saved(path))
                        })
                        .map_err(|error| error.to_string())
                };
                (id, result)
            }
            JobRequest::Cartoon {
                id,
                session_id,
                version,
                molecule,
                display,
                cancel,
            } => {
                let _ = (session_id, version);
                send_job_progress(&events, &window, id, "Building ribbon geometry", 0.2);
                let result = if cancel.load(Ordering::Relaxed) {
                    Ok(JobOutput::Cancelled)
                } else {
                    let prepared = prepare_cartoon(&molecule, &display);
                    if cancel.load(Ordering::Relaxed) {
                        Ok(JobOutput::Cancelled)
                    } else {
                        Ok(JobOutput::Cartoon(prepared))
                    }
                };
                (id, result)
            }
        };
        let _ = events.send(JobEvent::Complete {
            id,
            result: Box::new(result),
        });
        window.request_redraw();
    }
}

fn load_in_background(
    id: u64,
    _session_id: u64,
    _version: u64,
    path: &Path,
    cancel: &AtomicBool,
    events: &Sender<JobEvent>,
    window: &Window,
) -> std::result::Result<LoadedPayload, String> {
    send_job_progress(events, window, id, "Reading file", 0.1);
    let contents = read_local_file_limited(path).map_err(|error| error.to_string())?;
    if cancel.load(Ordering::Relaxed) {
        return Err("background operation canceled".into());
    }
    let filename = path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    send_job_progress(events, window, id, "Parsing structure", 0.45);
    if is_scene_document(&contents) {
        let document = decode_scene(&contents).map_err(|error| error.to_string())?;
        if cancel.load(Ordering::Relaxed) {
            return Err("background operation canceled".into());
        }
        send_job_progress(
            events,
            window,
            id,
            "Building hierarchy and spatial index",
            0.82,
        );
        let hierarchy = MoleculeHierarchy::from_molecule(&document.molecule);
        let atom_bvh = AtomBvh::build(&document.molecule);
        return Ok(LoadedPayload::Scene {
            filename,
            path: path.to_owned(),
            document: Box::new(document),
            hierarchy,
            atom_bvh,
        });
    }
    if path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case(SCENE_EXTENSION))
    {
        return Err(format!(
            "{} is not a Molecule 1.0 document (missing MOLECULE magic). MDL Molfile is not supported yet",
            path.display()
        ));
    }
    let (molecule, _) = parse_structure(&contents, &filename).map_err(|error| error.to_string())?;
    if cancel.load(Ordering::Relaxed) {
        return Err("background operation canceled".into());
    }
    send_job_progress(
        events,
        window,
        id,
        "Building hierarchy and spatial index",
        0.82,
    );
    let hierarchy = MoleculeHierarchy::from_molecule(&molecule);
    let atom_bvh = AtomBvh::build(&molecule);
    let display = DisplayState::for_molecule(&molecule);
    let molecule_id = molecule_id_from_structure(&contents, &filename);
    Ok(LoadedPayload::Structure {
        filename,
        molecule_id,
        molecule,
        hierarchy,
        atom_bvh,
        display: Box::new(display),
    })
}

fn send_job_progress(
    events: &Sender<JobEvent>,
    window: &Window,
    id: u64,
    stage: &'static str,
    progress: f32,
) {
    let _ = events.send(JobEvent::Progress {
        id,
        stage,
        progress,
    });
    window.request_redraw();
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("could not create {}", parent.display()))?;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("molecule.mol");
    let nonce = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(".{filename}.{}.{}.tmp", std::process::id(), nonce));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .with_context(|| format!("could not create {}", temporary.display()))?;
        file.write_all(contents)
            .with_context(|| format!("could not write {}", temporary.display()))?;
        file.flush()
            .with_context(|| format!("could not flush {}", temporary.display()))?;
        file.sync_all()
            .with_context(|| format!("could not sync {}", temporary.display()))?;
        drop(file);
        fs::rename(&temporary, path).with_context(|| {
            format!(
                "could not replace {} with {}",
                path.display(),
                temporary.display()
            )
        })?;
        if let Ok(directory) = OpenOptions::new().read(true).open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn recovery_directory() -> Result<PathBuf> {
    #[cfg(target_os = "macos")]
    let base = env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join("Library/Application Support"));
    #[cfg(target_os = "windows")]
    let base = env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let base = env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".local/state"))
        });
    let directory = base
        .context("could not determine the user data directory")?
        .join("molview/recovery");
    fs::create_dir_all(&directory)
        .with_context(|| format!("could not create {}", directory.display()))?;
    Ok(directory)
}

fn recovery_path_for(session_id: u64) -> Result<PathBuf> {
    Ok(recovery_directory()?.join(format!(
        "molview-{}-{session_id}.recovery.mol",
        std::process::id()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use molview::{
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
                EditableSnapshot {
                    display: None,
                    named_selections: BTreeMap::new(),
                    named_selection_expressions: BTreeMap::new(),
                    named_selection_styles: BTreeMap::new(),
                    measurement_lines: Vec::new(),
                    hierarchy_names: BTreeMap::new(),
                    inspection: Some(InspectionTarget::Atom(index)),
                    hierarchy_selection: BTreeSet::new(),
                    hierarchy_selection_anchor: None,
                },
            );
        }
        assert_eq!(history.len(), EDIT_HISTORY_LIMIT);
        assert_eq!(
            history.front().and_then(|state| state.inspection),
            Some(InspectionTarget::Atom(10))
        );
        assert_eq!(
            history.back().and_then(|state| state.inspection),
            Some(InspectionTarget::Atom(59))
        );
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
            "molview-p0-{}-{}",
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
        let directory = env::temp_dir().join(format!("molview-fetch-test-{}", std::process::id()));
        let path = download_pdb("4R8P", &directory).unwrap();
        let contents = fs::read(&path).unwrap();
        let (molecule, _) = parse_structure(&contents, "4R8P.cif").unwrap();
        assert!(!molecule.atoms.is_empty());
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}

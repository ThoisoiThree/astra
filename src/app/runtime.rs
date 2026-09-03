use super::*;

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

pub(super) struct Runtime {
    pub(super) window: Arc<Window>,
    pub(super) renderer: Renderer,
    pub(super) egui_context: egui::Context,
    pub(super) egui_state: egui_winit::State,
    pub(super) molecule: Option<Molecule>,
    pub(super) hierarchy: Option<MoleculeHierarchy>,
    pub(super) atom_bvh: Option<AtomBvh>,
    pub(super) display: Option<DisplayState>,
    pub(super) named_selections: BTreeMap<String, Selection>,
    pub(super) named_selection_expressions: BTreeMap<String, String>,
    pub(super) named_selection_styles: BTreeMap<String, NamedSelectionStyle>,
    pub(super) named_selection_statuses: BTreeMap<String, SelectionStatus>,
    pub(super) measurement_lines: Vec<MeasurementLine>,
    pub(super) next_measurement_id: u64,
    pub(super) hierarchy_names: BTreeMap<InspectionTarget, String>,
    pub(super) inspection: Option<InspectionTarget>,
    pub(super) hierarchy_selection: BTreeSet<InspectionTarget>,
    pub(super) hierarchy_selection_anchor: Option<InspectionTarget>,
    pub(super) focus_description: String,
    pub(super) pivot_description: String,
    pub(super) loaded_filename: Option<String>,
    pub(super) molecule_id: Option<String>,
    pub(super) scene_path: Option<PathBuf>,
    pub(super) dirty: bool,
    pub(super) recovery_path: Option<PathBuf>,
    pub(super) autosave_due: Option<Instant>,
    pub(super) document_version: u64,
    pub(super) needs_cartoon_refresh: bool,
    pub(super) next_pick_request_id: u64,
    pub(super) pending_pick: Option<PendingPick>,
    pub(super) inactive_sessions: Vec<DocumentSession>,
    pub(super) session_order: Vec<u64>,
    pub(super) active_session_id: u64,
    pub(super) next_session_id: u64,
    pub(super) fetch_receiver: Option<Receiver<FetchEvent>>,
    pub(super) fetching_pdb_id: Option<String>,
    pub(super) fetch_progress: FetchProgress,
    pub(super) fetch_cancel: Option<Arc<AtomicBool>>,
    pub(super) job_sender: SyncSender<JobRequest>,
    pub(super) job_receiver: Receiver<JobEvent>,
    pub(super) background_jobs: BTreeMap<u64, BackgroundJob>,
    pub(super) next_job_id: u64,
    pub(super) camera: OrbitCamera,
    pub(super) ui: UiState,
    pub(super) cursor: Option<PhysicalPosition<f64>>,
    pub(super) viewport: Viewport,
    pub(super) left_drag: bool,
    pub(super) left_drag_distance: f32,
    pub(super) right_drag: bool,
    pub(super) modifiers: ModifiersState,
    pub(super) focused: bool,
    pub(super) occluded: bool,
    pub(super) undo_history: VecDeque<EditOperation>,
    pub(super) redo_history: VecDeque<EditOperation>,
}

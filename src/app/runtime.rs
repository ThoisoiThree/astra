use super::*;

pub fn run(options: crate::cli::Options) -> Result<()> {
    let event_loop = EventLoop::new().context("could not create the native event loop")?;
    let mut application = AstraApplication {
        runtime: None,
        options: Some(options),
        startup_error: None,
    };
    event_loop
        .run_app(&mut application)
        .context("native event loop failed")?;
    if let Some(error) = application.startup_error {
        return Err(error);
    }
    if let Some(error) = application
        .runtime
        .as_mut()
        .and_then(|runtime| runtime.batch_failure.take())
    {
        anyhow::bail!(error);
    }
    Ok(())
}

struct AstraApplication {
    runtime: Option<Runtime>,
    options: Option<crate::cli::Options>,
    startup_error: Option<anyhow::Error>,
}

impl ApplicationHandler for AstraApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.runtime.is_some() {
            return;
        }
        match Runtime::new(event_loop, self.options.take().unwrap_or_default()) {
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
            runtime.poll_pick_result();
            if runtime.batch.is_some() {
                runtime.advance_batch();
                if runtime.batch.is_none() {
                    event_loop.exit();
                    return;
                }
                // Background work completes on another thread; poll while rendering in batch.
                event_loop.set_control_flow(ControlFlow::WaitUntil(
                    Instant::now() + Duration::from_millis(20),
                ));
                return;
            }
            if runtime.exit_ready {
                event_loop.exit();
                return;
            }
            runtime.autosave_due_documents();
            let now = Instant::now();
            let visible = (runtime.focused || runtime.pending_close.is_some()) && !runtime.occluded;
            if visible && runtime.repaint_due.is_some_and(|due| due <= now) {
                runtime.repaint_due = None;
                runtime.window.request_redraw();
            }
            let repaint = visible.then_some(runtime.repaint_due).flatten();
            // Readback needs device polling even when the pointer no longer moves.
            let pick = runtime
                .pending_pick
                .as_ref()
                .map(|_| now + Duration::from_millis(8));
            let deadline = [runtime.next_autosave_deadline(), repaint, pick]
                .into_iter()
                .flatten()
                .min();
            if let Some(deadline) = deadline {
                event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
            } else {
                event_loop.set_control_flow(ControlFlow::Wait);
            }
        }
    }
}

pub(super) struct Runtime {
    #[cfg(target_os = "windows")]
    pub(super) windows_backend: astra::render::backend::WindowsBackend,
    pub(super) window: Arc<Window>,
    pub(super) renderer: Renderer,
    pub(super) egui_context: egui::Context,
    pub(super) egui_state: egui_winit::State,
    pub(super) molecule: Option<Molecule>,
    pub(super) hierarchy: Option<MoleculeHierarchy>,
    pub(super) secondary_structure: Option<Vec<Vec<SecondaryStructure>>>,
    pub(super) atom_bvh: Option<AtomBvh>,
    pub(super) display: Option<DisplayState>,
    pub(super) named_selections: BTreeMap<String, Selection>,
    pub(super) named_selection_expressions: BTreeMap<String, String>,
    pub(super) named_selection_styles: BTreeMap<String, NamedSelectionStyle>,
    pub(super) named_selection_statuses: BTreeMap<String, SelectionStatus>,
    pub(super) measurement_lines: Vec<MeasurementLine>,
    /// Derived from the display state; refreshed whenever geometry is rebuilt.
    pub(super) label_items: Vec<astra::labels::LabelItem>,
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
    pub(super) cartoon_generation: u64,
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
    pub(super) repaint_due: Option<Instant>,
    pub(super) pending_close: Option<closing::ClosePlan>,
    pub(super) exit_ready: bool,
    pub(super) recovery_scan_pending: bool,
    pub(super) recovery_candidates: VecDeque<PathBuf>,
    /// Command-line batch rendering in progress.
    pub(super) batch: Option<super::structure::BatchState>,
    pub(super) batch_failure: Option<String>,
}

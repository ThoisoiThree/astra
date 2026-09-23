//! Trajectory playback. A worker thread owns the frame reader; the render thread asks for
//! frames by index and applies each one by uploading positions only. Ribbons, surfaces
//! and labels follow asynchronously, dropping intermediate frames when they fall behind.

use std::{
    sync::mpsc::{self, Receiver, Sender, TryRecvError},
    thread,
};

use astra::molecule::trajectory::{
    Frame, FrameReader, InMemoryFrames, PlaybackSettings, TrajectoryFormat, open_trajectory,
};

use super::*;

enum WorkerEvent {
    Opened {
        format: TrajectoryFormat,
        frame_count: usize,
    },
    Failed(String),
    Frame(usize, Result<Frame, String>),
}

pub(super) struct TrajectoryPlayer {
    /// Shown in the timeline: file name or "N models".
    pub(super) label: String,
    /// Trajectory file, or `None` for models held in memory.
    pub(super) path: Option<PathBuf>,
    /// In-memory models, kept so scenes can store them.
    pub(super) models: Option<Arc<Vec<Vec<Vec3>>>>,
    pub(super) format: Option<TrajectoryFormat>,
    /// Zero while the file is being indexed.
    pub(super) frame_count: usize,
    pub(super) current: usize,
    pub(super) playing: bool,
    pub(super) settings: PlaybackSettings,
    pub(super) time: Option<f64>,
    pub(super) cell: Option<[f64; 6]>,
    /// Coordinates of the structure before the trajectory was loaded, restored on unload.
    pub(super) reference_positions: Vec<Vec3>,
    pending: Option<usize>,
    /// Frame to show once the file is indexed.
    initial_frame: Option<usize>,
    direction: i8,
    last_step: Instant,
    requests: Sender<usize>,
    events: Receiver<WorkerEvent>,
}

impl TrajectoryPlayer {
    /// Opens a trajectory file on a worker thread; frames are indexed there.
    pub(super) fn open_file(
        path: PathBuf,
        atom_count: usize,
        reference_positions: Vec<Vec3>,
        window: Arc<Window>,
    ) -> Self {
        let label = path.file_name().map_or_else(
            || path.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        );
        let opened_path = path.clone();
        let player = Self::spawn(
            label,
            Some(path),
            None,
            reference_positions,
            window,
            move || open_trajectory(&opened_path, atom_count).map_err(|error| error.to_string()),
        );
        log::info!("Opening trajectory {}", player.label);
        player
    }

    /// Plays models already in memory, for example an NMR ensemble.
    pub(super) fn from_models(
        models: Vec<Vec<Vec3>>,
        reference_positions: Vec<Vec3>,
        window: Arc<Window>,
    ) -> Self {
        let models = Arc::new(models);
        let label = format!("{} models", models.len());
        let source = models.clone();
        Self::spawn(
            label,
            None,
            Some(models),
            reference_positions,
            window,
            move || {
                InMemoryFrames::new(source.iter().cloned().map(Frame::new).collect())
                    .map(|frames| Box::new(frames) as Box<dyn FrameReader>)
                    .map_err(|error| error.to_string())
            },
        )
    }

    fn spawn(
        label: String,
        path: Option<PathBuf>,
        models: Option<Arc<Vec<Vec<Vec3>>>>,
        reference_positions: Vec<Vec3>,
        window: Arc<Window>,
        open: impl FnOnce() -> Result<Box<dyn FrameReader>, String> + Send + 'static,
    ) -> Self {
        let (request_sender, request_receiver) = mpsc::channel::<usize>();
        let (event_sender, event_receiver) = mpsc::channel();
        let spawned = thread::Builder::new()
            .name("trajectory reader".into())
            .spawn(move || {
                let send = |event| {
                    let delivered = event_sender.send(event).is_ok();
                    window.request_redraw();
                    delivered
                };
                let mut reader = match open() {
                    Ok(reader) => reader,
                    Err(error) => {
                        send(WorkerEvent::Failed(error));
                        return;
                    }
                };
                if !send(WorkerEvent::Opened {
                    format: reader.format(),
                    frame_count: reader.len(),
                }) {
                    return;
                }
                // Only the newest request matters when the user scrubs faster than frames
                // can be read.
                while let Ok(mut index) = request_receiver.recv() {
                    loop {
                        match request_receiver.try_recv() {
                            Ok(newer) => index = newer,
                            Err(TryRecvError::Empty) => break,
                            Err(TryRecvError::Disconnected) => return,
                        }
                    }
                    let frame = reader.read_frame(index).map_err(|error| error.to_string());
                    if !send(WorkerEvent::Frame(index, frame)) {
                        return;
                    }
                }
            });
        if let Err(error) = spawned {
            log::error!("could not start the trajectory reader: {error}");
        }
        Self {
            label,
            path,
            models,
            format: None,
            frame_count: 0,
            current: 0,
            playing: false,
            settings: PlaybackSettings::default(),
            time: None,
            cell: None,
            reference_positions,
            pending: None,
            initial_frame: None,
            direction: 1,
            last_step: Instant::now(),
            requests: request_sender,
            events: event_receiver,
        }
    }

    /// Recreates a trajectory saved in a scene. Relative paths resolve against the scene.
    pub(super) fn restore(
        saved: astra::scene::SceneTrajectory,
        scene_path: Option<&Path>,
        window: Arc<Window>,
    ) -> Self {
        let reference = saved.reference_positions;
        let mut player = match saved.path {
            Some(path) => {
                let mut path = PathBuf::from(path);
                if path.is_relative()
                    && let Some(directory) = scene_path.and_then(Path::parent)
                {
                    path = directory.join(path);
                }
                let atom_count = reference.len();
                Self::open_file(path, atom_count, reference, window)
            }
            None => Self::from_models(saved.models, reference, window),
        };
        player.settings = saved.playback.sanitized();
        player.seek(saved.frame);
        player
    }

    pub(super) fn scene_trajectory(&self) -> astra::scene::SceneTrajectory {
        astra::scene::SceneTrajectory {
            path: self
                .path
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned()),
            models: self
                .models
                .as_ref()
                .map(|models| models.as_ref().clone())
                .unwrap_or_default(),
            reference_positions: self.reference_positions.clone(),
            frame: self.current,
            playback: self.settings,
        }
    }

    pub(super) fn is_ready(&self) -> bool {
        self.format.is_some()
    }

    /// True while a requested frame has not arrived yet (or the file is being indexed).
    pub(super) fn is_busy(&self) -> bool {
        !self.is_ready() || self.pending.is_some() || self.initial_frame.is_some()
    }

    /// Shows `index` as soon as it is read; before indexing finishes it is remembered.
    pub(super) fn seek(&mut self, index: usize) {
        if !self.is_ready() {
            self.initial_frame = Some(index);
            return;
        }
        let index = index.min(self.frame_count.saturating_sub(1));
        self.pending = Some(index);
        let _ = self.requests.send(index);
    }

    pub(super) fn step(&mut self, delta: i64) {
        self.playing = false;
        let base = self.pending.unwrap_or(self.current) as i64;
        let last = self.frame_count.saturating_sub(1) as i64;
        self.seek((base + delta).clamp(0, last) as usize);
    }

    pub(super) fn set_playing(&mut self, playing: bool) {
        self.playing = playing && self.frame_count > 1;
        if self.playing
            && self.settings.loop_mode == astra::molecule::trajectory::LoopMode::Once
            && self.current + 1 >= self.frame_count
        {
            // Replaying a finished one-shot run starts from the beginning.
            self.direction = 1;
            self.seek(0);
        }
        self.last_step = Instant::now();
    }

    /// When the next frame is due during playback.
    pub(super) fn deadline(&self) -> Option<Instant> {
        (self.playing && self.pending.is_none())
            .then(|| self.last_step + Duration::from_secs_f32(1.0 / self.settings.fps.max(1.0)))
    }
}

impl Runtime {
    pub(super) fn scene_trajectory(&self) -> Option<astra::scene::SceneTrajectory> {
        self.trajectory
            .as_ref()
            .map(TrajectoryPlayer::scene_trajectory)
    }

    /// Loads a trajectory file for the active structure.
    pub(super) fn load_trajectory(&mut self, path: PathBuf) -> Result<()> {
        let molecule = self
            .molecule
            .as_ref()
            .context("open a structure before loading a trajectory")?;
        let reference = self.trajectory.as_ref().map_or_else(
            || molecule.positions(),
            |player| player.reference_positions.clone(),
        );
        self.trajectory = Some(TrajectoryPlayer::open_file(
            path,
            molecule.atoms.len(),
            reference,
            self.window.clone(),
        ));
        if let Some(player) = &mut self.trajectory {
            player.seek(0);
        }
        self.mark_dirty();
        Ok(())
    }

    /// Asks for a trajectory file and loads it.
    pub(super) fn load_trajectory_dialog(&mut self) -> Result<()> {
        if self.molecule.is_none() {
            anyhow::bail!("open a structure before loading a trajectory");
        }
        let mut extensions: Vec<&str> = TrajectoryFormat::EXTENSIONS.to_vec();
        extensions.extend(["pdb", "ent", "cif", "mmcif", "gro", "xyz"]);
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Trajectories and multi-model structures", &extensions)
            .add_filter("All files", &["*"])
            .pick_file()
        else {
            return Ok(());
        };
        self.load_trajectory(path)
    }

    /// Removes the trajectory and restores the structure's own coordinates.
    pub(super) fn unload_trajectory(&mut self) {
        let Some(player) = self.trajectory.take() else {
            return;
        };
        if let Some(molecule) = &mut self.molecule
            && molecule.set_positions(&player.reference_positions)
        {
            self.renderer.update_positions(molecule);
            self.after_positions_changed();
        }
        self.mark_dirty();
    }

    pub(super) fn handle_trajectory_action(&mut self, action: crate::ui::TrajectoryAction) {
        use crate::ui::TrajectoryAction;
        if matches!(action, TrajectoryAction::Load) {
            if let Err(error) = self.load_trajectory_dialog() {
                self.ui.latest_error = Some(format!("{error:#}"));
            }
            return;
        }
        if matches!(action, TrajectoryAction::Unload) {
            self.unload_trajectory();
            return;
        }
        let Some(player) = &mut self.trajectory else {
            return;
        };
        match action {
            TrajectoryAction::Play => player.set_playing(true),
            TrajectoryAction::Pause => player.playing = false,
            TrajectoryAction::Seek(index) => {
                player.playing = false;
                player.seek(index);
            }
            TrajectoryAction::Step(delta) => player.step(delta),
            TrajectoryAction::Settings(settings) => player.settings = settings.sanitized(),
            TrajectoryAction::Load | TrajectoryAction::Unload => {}
        }
        self.window.request_redraw();
    }

    /// Applies frames from the reader and advances playback.
    pub(super) fn poll_trajectory(&mut self) {
        let atom_count = self.molecule.as_ref().map(|molecule| molecule.atoms.len());
        let Some(player) = &mut self.trajectory else {
            return;
        };
        if atom_count != Some(player.reference_positions.len()) {
            // The topology changed (for example after adding hydrogens).
            log::info!(
                "Trajectory {} unloaded: the atom count changed",
                player.label
            );
            self.trajectory = None;
            return;
        }
        let mut arrived = None;
        loop {
            match player.events.try_recv() {
                Ok(WorkerEvent::Opened {
                    format,
                    frame_count,
                }) => {
                    log::info!(
                        "Trajectory {}: {} frames ({})",
                        player.label,
                        frame_count,
                        format.label()
                    );
                    player.format = Some(format);
                    player.frame_count = frame_count;
                    if let Some(index) = player.initial_frame.take() {
                        player.seek(index);
                    }
                }
                Ok(WorkerEvent::Failed(error)) => {
                    let label = player.label.clone();
                    self.trajectory = None;
                    self.ui.latest_error = Some(format!("Could not load {label}: {error}"));
                    return;
                }
                Ok(WorkerEvent::Frame(index, result)) => {
                    if player.pending == Some(index) {
                        player.pending = None;
                        match result {
                            Ok(frame) => arrived = Some((index, frame)),
                            Err(error) => {
                                player.playing = false;
                                self.ui.latest_error =
                                    Some(format!("Could not read frame {}: {error}", index + 1));
                            }
                        }
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if !player.is_ready() {
                        self.trajectory = None;
                    }
                    break;
                }
            }
        }
        if let Some((index, frame)) = arrived {
            self.apply_trajectory_frame(index, frame);
        }
        let Some(player) = &mut self.trajectory else {
            return;
        };
        if player.deadline().is_some_and(|due| due <= Instant::now()) {
            match player
                .settings
                .next_frame(player.current, player.frame_count, player.direction)
            {
                Some((next, direction)) => {
                    player.direction = direction;
                    player.last_step = Instant::now();
                    player.seek(next);
                }
                None => player.playing = false,
            }
        }
    }

    fn apply_trajectory_frame(&mut self, index: usize, frame: Frame) {
        let Some(molecule) = &mut self.molecule else {
            return;
        };
        if !molecule.set_positions(&frame.positions) {
            self.ui.latest_error = Some(format!(
                "frame {} has {} atoms but the structure has {}",
                index + 1,
                frame.positions.len(),
                molecule.atoms.len()
            ));
            self.trajectory = None;
            return;
        }
        self.renderer.update_positions(molecule);
        if let Some(player) = &mut self.trajectory {
            player.current = index;
            player.time = frame.time;
            player.cell = frame.cell;
        }
        self.after_positions_changed();
    }

    /// Everything derived from coordinates except the GPU positions themselves.
    fn after_positions_changed(&mut self) {
        // The picking hierarchy is rebuilt on the next CPU pick.
        self.atom_bvh = None;
        self.refresh_labels();
        self.request_frame_geometry();
        self.window.request_redraw();
    }
}

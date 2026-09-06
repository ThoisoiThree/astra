use super::*;

pub(super) enum CloseSave {
    Idle,
    ChoosingPath {
        session: u64,
        result: Receiver<Option<PathBuf>>,
    },
    Writing(u64),
}

pub(super) struct ClosePlan {
    targets: Vec<u64>,
    exit: bool,
    original_session: u64,
    // A discard approves one document revision, not future changes to that tab.
    discarded: BTreeMap<u64, u64>,
    prompt: Option<u64>,
    save: CloseSave,
}

impl ClosePlan {
    fn new(targets: Vec<u64>, exit: bool, original_session: u64) -> Self {
        Self {
            targets,
            exit,
            original_session,
            discarded: BTreeMap::new(),
            prompt: None,
            save: CloseSave::Idle,
        }
    }

    pub(super) fn is_busy(&self) -> bool {
        !matches!(self.save, CloseSave::Idle)
    }

    fn needs_confirmation(&self, id: u64, dirty: bool, version: u64) -> bool {
        self.targets.contains(&id) && dirty && self.discarded.get(&id) != Some(&version)
    }
}

impl Runtime {
    fn begin_close(&mut self, targets: Vec<u64>, exit: bool) {
        self.left_drag = false;
        self.right_drag = false;
        if self.pending_close.is_none() {
            self.pending_close = Some(ClosePlan::new(targets, exit, self.active_session_id));
            self.advance_close();
        }
        self.window.request_redraw();
    }

    pub(super) fn request_close_session(&mut self, id: u64) {
        if self.session_order.contains(&id) {
            self.begin_close(vec![id], false);
        }
    }

    pub(super) fn request_exit(&mut self) -> bool {
        self.begin_close(self.session_order.clone(), true);
        self.exit_ready
    }

    pub(super) fn handle_close_action(&mut self, action: CloseAction) {
        let Some(mut plan) = self.pending_close.take() else {
            return;
        };
        if action == CloseAction::Cancel {
            self.activate_session(plan.original_session);
            self.window.request_redraw();
            return;
        }
        if plan.is_busy() {
            self.pending_close = Some(plan);
            return;
        }
        let Some(session) = plan.prompt else {
            self.pending_close = Some(plan);
            return;
        };
        self.activate_session(session);
        match action {
            CloseAction::Discard => {
                plan.discarded.insert(session, self.document_version);
            }
            CloseAction::Save => {
                self.ui.latest_error = None;
                if let Some(path) = self.scene_path.clone() {
                    let job_id = self.next_job_id;
                    match self.save_scene_to(path) {
                        Ok(()) => plan.save = CloseSave::Writing(job_id),
                        Err(error) => self.ui.latest_error = Some(error.to_string()),
                    }
                } else {
                    let suggested = self
                        .loaded_filename
                        .as_deref()
                        .and_then(|name| Path::new(name).file_stem())
                        .map_or_else(
                            || "scene.mol".into(),
                            |stem| format!("{}.mol", stem.to_string_lossy()),
                        );
                    let dialog = rfd::AsyncFileDialog::new()
                        .set_parent(self.window.as_ref())
                        .add_filter(SCENE_FORMAT_NAME, &[SCENE_EXTENSION])
                        .set_file_name(suggested);
                    let (sender, result) = mpsc::channel();
                    let window = self.window.clone();
                    // Keep pumping the owner window while the native file picker is open.
                    thread::spawn(move || {
                        let path = pollster::block_on(dialog.save_file())
                            .map(|file| file.path().to_owned());
                        let _ = sender.send(path);
                        window.request_redraw();
                    });
                    plan.save = CloseSave::ChoosingPath { session, result };
                }
            }
            CloseAction::Cancel => {}
        }
        self.pending_close = Some(plan);
        self.advance_close();
        self.window.request_redraw();
    }

    pub(super) fn advance_close(&mut self) {
        let Some(mut plan) = self.pending_close.take() else {
            return;
        };
        match &plan.save {
            CloseSave::ChoosingPath { session, result } => match result.try_recv() {
                Err(TryRecvError::Empty) => {
                    self.pending_close = Some(plan);
                    return;
                }
                Ok(None) => {
                    self.activate_session(plan.original_session);
                    self.window.request_redraw();
                    return;
                }
                Ok(Some(path)) => {
                    self.activate_session(*session);
                    let job_id = self.next_job_id;
                    plan.save = match self.save_scene_to(path) {
                        Ok(()) => CloseSave::Writing(job_id),
                        Err(error) => {
                            self.ui.latest_error = Some(error.to_string());
                            CloseSave::Idle
                        }
                    };
                    self.pending_close = Some(plan);
                    self.window.request_redraw();
                    return;
                }
                Err(TryRecvError::Disconnected) => {
                    self.ui.latest_error =
                        Some("Save dialog stopped unexpectedly; the scene remains open".into());
                    plan.save = CloseSave::Idle;
                }
            },
            CloseSave::Writing(job) if self.background_jobs.contains_key(job) => {
                self.pending_close = Some(plan);
                return;
            }
            CloseSave::Writing(_) => plan.save = CloseSave::Idle,
            CloseSave::Idle => {}
        }

        // Do not destroy the window/process while an already requested save is pending.
        if let Some((&job, work)) = self
            .background_jobs
            .iter()
            .find(|(_, job)| job.kind == JobKind::Save && plan.targets.contains(&job.session_id))
        {
            let session = work.session_id;
            plan.save = CloseSave::Writing(job);
            plan.prompt = Some(session);
            self.activate_session(session);
            self.pending_close = Some(plan);
            self.window.request_redraw();
            return;
        }
        let dirty = plan.targets.iter().copied().find(|id| {
            if *id == self.active_session_id {
                plan.needs_confirmation(*id, self.dirty, self.document_version)
            } else {
                self.inactive_sessions
                    .iter()
                    .find(|session| session.id == *id)
                    .is_some_and(|session| {
                        plan.needs_confirmation(*id, session.dirty, session.document_version)
                    })
            }
        });
        if let Some(id) = dirty {
            if plan.prompt != Some(id) {
                self.activate_session(id);
                plan.prompt = Some(id);
                self.window.request_redraw();
            }
            self.pending_close = Some(plan);
            return;
        }

        // Commit only after all decisions. Cancel must leave earlier tabs and
        // their recovery files intact, even if Discard was chosen for one of them.
        for id in &plan.targets {
            for job in self
                .background_jobs
                .values()
                .filter(|job| job.session_id == *id)
            {
                job.cancel.store(true, Ordering::Relaxed);
            }
            if *id == self.active_session_id {
                self.remove_recovery_file();
            } else if let Some(session) = self
                .inactive_sessions
                .iter_mut()
                .find(|session| session.id == *id)
                && let Some(path) = session.recovery_path.take()
            {
                let _ = fs::remove_file(path);
            }
        }
        if plan.exit {
            self.exit_ready = true;
        } else if let Some(id) = plan.targets.first() {
            self.close_session(*id);
        }
        self.window.request_redraw();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discard_is_scoped_to_the_confirmed_document_revision() {
        let mut plan = ClosePlan::new(vec![1, 2], true, 1);
        assert!(!plan.needs_confirmation(1, false, 3));
        assert!(plan.needs_confirmation(1, true, 3));
        plan.discarded.insert(1, 3);
        assert!(!plan.needs_confirmation(1, true, 3));
        assert!(plan.needs_confirmation(1, true, 4));
        assert!(plan.needs_confirmation(2, true, 3));
        assert!(!plan.needs_confirmation(9, true, 3));
    }

    #[test]
    fn a_failed_save_does_not_approve_discarding_changes() {
        let mut plan = ClosePlan::new(vec![1], false, 1);
        plan.save = CloseSave::Writing(42);
        assert!(plan.is_busy());
        plan.save = CloseSave::Idle;
        // On completion, dirty stays true for failures and stale-version writes.
        assert!(plan.needs_confirmation(1, true, 4));
        assert!(!plan.needs_confirmation(1, false, 4));
    }
}

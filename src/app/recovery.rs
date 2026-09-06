use super::*;

/// Discovery does not read or decode scene contents. Corrupt or large autosaves
/// must not prevent the first frame or require loading every scene into memory.
pub(super) fn scan_files(directory: &Path) -> Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error).context("could not scan recovery directory"),
    };
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.context("could not inspect recovery directory entry")?;
        if entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.ends_with(".recovery.mol"))
            && entry
                .file_type()
                .context("could not inspect recovery file type")?
                .is_file()
        {
            paths.push(entry.path());
        }
    }
    paths.sort();
    Ok(paths)
}

impl Runtime {
    pub(super) fn start_recovery_scan_after_frame(&mut self) {
        if !self.recovery_scan_pending {
            return;
        }
        match self.job_sender.try_send(JobRequest::ScanRecovery) {
            Ok(()) => self.recovery_scan_pending = false,
            // Existing jobs wake the window on completion, so retry after that frame.
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => {
                self.recovery_scan_pending = false;
                self.ui.latest_error =
                    Some("could not scan autosaves: background worker stopped".into());
            }
        }
    }

    pub(super) fn handle_recovery_action(&mut self, action: RecoveryAction) {
        if action == RecoveryAction::Later {
            self.recovery_candidates.clear();
            self.window.request_redraw();
            return;
        }
        let Some(path) = self.recovery_candidates.front().cloned() else {
            return;
        };
        let result = match action {
            RecoveryAction::Restore => self.start_load_path_with_kind(&path, JobKind::Recover),
            RecoveryAction::Discard => self.submit_job(JobRequest::DiscardRecovery { path }),
            RecoveryAction::Later => return,
        };
        match result {
            Ok(()) => {
                self.recovery_candidates.pop_front();
            }
            Err(error) => self.ui.latest_error = Some(error.to_string()),
        }
        self.window.request_redraw();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovery_discovery_keeps_corrupt_files_without_decoding_them() -> Result<()> {
        let directory = env::temp_dir().join(format!(
            "astra-recovery-scan-{}-{}",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory)?;
        let first = directory.join("a.recovery.mol");
        let second = directory.join("b.recovery.mol");
        fs::write(&first, b"corrupt scene")?;
        fs::write(&second, b"")?;
        fs::write(directory.join("ordinary.mol"), b"not an autosave")?;
        fs::create_dir(directory.join("folder.recovery.mol"))?;
        assert_eq!(scan_files(&directory)?, vec![first.clone(), second]);
        assert_eq!(fs::read(first)?, b"corrupt scene");
        assert!(scan_files(&directory.join("missing"))?.is_empty());
        fs::remove_dir_all(directory)?;
        Ok(())
    }
}

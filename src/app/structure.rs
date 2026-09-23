//! Image export and structures derived from the active document.

use super::*;

use crate::ui::{ExportRequest, StructureRequest};

impl Runtime {
    /// Asks for a PNG path and exports the current view. Returns the written path, or `None`
    /// when the dialog was dismissed.
    pub(super) fn export_image_dialog(
        &mut self,
        request: ExportRequest,
    ) -> Result<Option<PathBuf>> {
        let suggested = format!(
            "{}.png",
            session_label(self.molecule_id.as_deref(), self.loaded_filename.as_deref())
        );
        let Some(path) = rfd::FileDialog::new()
            .add_filter("PNG image", &["png"])
            .set_file_name(suggested)
            .save_file()
        else {
            return Ok(None);
        };
        let path = if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
        {
            path
        } else {
            path.with_extension("png")
        };
        self.export_image_to(&path, request)?;
        Ok(Some(path))
    }

    pub(super) fn export_image_to(&mut self, path: &Path, request: ExportRequest) -> Result<()> {
        if self.molecule.is_none() {
            anyhow::bail!("open a structure before exporting an image");
        }
        let image = self
            .renderer
            .render_image(&self.camera, self.display.as_ref(), &request.image)
            .context("could not render the image")?;
        if image.supersampling < request.image.supersampling {
            log::info!(
                "Export supersampling reduced from {}× to {}× by the GPU texture limit",
                request.image.supersampling,
                image.supersampling
            );
        }
        let bytes = astra::image_export::encode_png(&image, request.dots_per_inch)
            .context("could not encode PNG")?;
        atomic_write(path, &bytes)
            .with_context(|| format!("could not write {}", path.display()))?;
        self.window.request_redraw();
        Ok(())
    }

    /// Builds an assembly or crystal packing in the background and opens it in a new tab.
    pub(super) fn start_derived_structure(&mut self, request: StructureRequest) -> Result<()> {
        let molecule = self
            .molecule
            .clone()
            .context("open a structure before building an assembly")?;
        let secondary_source = self
            .display
            .as_ref()
            .map_or_else(Default::default, |display| display.secondary_source);
        let base = session_label(self.molecule_id.as_deref(), self.loaded_filename.as_deref());
        let label = format!("{base} · {}", request.label());
        self.begin_new_session();
        self.clear_active_document();
        self.loaded_filename = Some(label.clone());
        self.molecule_id = Some(label.clone());
        self.document_version = self.document_version.wrapping_add(1);
        let id = self.next_job_id;
        self.next_job_id = self.next_job_id.wrapping_add(1);
        let cancel = Arc::new(AtomicBool::new(false));
        self.submit_job(JobRequest::Derive {
            id,
            molecule: Box::new(molecule),
            request,
            label,
            secondary_source,
            cancel: cancel.clone(),
        })?;
        self.background_jobs.insert(
            id,
            BackgroundJob {
                session_id: self.active_session_id,
                version: self.document_version,
                kind: JobKind::Load,
                stage: "Queued structure generation".into(),
                progress: 0.0,
                cancel,
            },
        );
        Ok(())
    }
}

/// Builds the requested structure; runs on the background worker.
pub(super) fn derive_structure(
    molecule: &Molecule,
    request: &StructureRequest,
) -> std::result::Result<Molecule, String> {
    use astra::molecule::symmetry;
    match request {
        StructureRequest::Assembly(id) => symmetry::build_assembly(molecule, id),
        StructureRequest::UnitCell => symmetry::build_unit_cell(molecule),
        StructureRequest::SymmetryMates(radius) => {
            symmetry::build_symmetry_mates(molecule, *radius)
        }
    }
    .map_err(|error| error.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchStage {
    Loading,
    Deriving,
    Configuring,
    Rendering,
}

/// Command-line rendering: load, optionally build an assembly, set the mode, export, exit.
pub(super) struct BatchState {
    export: crate::cli::BatchExport,
    stage: BatchStage,
}

impl BatchState {
    pub(super) fn new(export: crate::cli::BatchExport) -> Self {
        Self {
            export,
            stage: BatchStage::Loading,
        }
    }
}

impl Runtime {
    /// Advances batch rendering once background work is idle; clears `batch` when done.
    pub(super) fn advance_batch(&mut self) {
        let Some(batch) = &self.batch else {
            return;
        };
        if !self.background_jobs.is_empty() || self.needs_cartoon_refresh {
            return;
        }
        if self.molecule.is_none() {
            let error = self
                .ui
                .latest_error
                .clone()
                .unwrap_or_else(|| "the structure could not be loaded".into());
            self.finish_batch(Err(error));
            return;
        }
        match batch.stage {
            BatchStage::Loading => {
                let assembly = batch.export.assembly.clone();
                self.set_batch_stage(BatchStage::Deriving);
                if let Some(id) = assembly
                    && let Err(error) = self.start_derived_structure(StructureRequest::Assembly(id))
                {
                    self.finish_batch(Err(format!("{error:#}")));
                }
            }
            BatchStage::Deriving => {
                let mode = batch.export.mode;
                self.set_batch_stage(BatchStage::Configuring);
                if let Some(mode) = mode {
                    self.handle_manager_action(crate::ui::ManagerAction::SetGlobalMode(mode));
                }
            }
            BatchStage::Configuring => self.set_batch_stage(BatchStage::Rendering),
            BatchStage::Rendering => {
                let (output, request) = (batch.export.output.clone(), batch.export.request);
                let result = self
                    .export_image_to(&output, request)
                    .map(|()| output)
                    .map_err(|error| format!("{error:#}"));
                self.finish_batch(result.map(|output| {
                    println!("{}", output.display());
                }));
            }
        }
    }

    fn set_batch_stage(&mut self, stage: BatchStage) {
        if let Some(batch) = &mut self.batch {
            batch.stage = stage;
        }
    }

    fn finish_batch(&mut self, result: std::result::Result<(), String>) {
        if let Err(error) = result {
            log::error!("Batch rendering failed: {error}");
            self.batch_failure = Some(error);
        }
        self.batch = None;
    }
}

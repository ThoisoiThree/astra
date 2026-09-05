use super::*;

pub(super) struct DocumentSession {
    pub(super) id: u64,
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
    pub(super) camera: OrbitCamera,
    pub(super) undo_history: VecDeque<EditOperation>,
    pub(super) redo_history: VecDeque<EditOperation>,
}

impl DocumentSession {
    pub(super) fn empty(id: u64, viewport: Viewport) -> Self {
        let aspect = viewport.width / viewport.height.max(1.0);
        Self {
            id,
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
            camera: OrbitCamera::new(aspect),
            undo_history: VecDeque::new(),
            redo_history: VecDeque::new(),
        }
    }

    pub(super) fn tab(&self) -> SessionTab {
        SessionTab {
            id: self.id,
            label: session_label(self.molecule_id.as_deref(), self.loaded_filename.as_deref()),
            dirty: self.dirty,
        }
    }

    pub(super) fn write_scene(&self, path: &Path) -> Result<()> {
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

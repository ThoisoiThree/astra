use super::*;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct FetchProgress {
    pub(super) downloaded_bytes: u64,
    pub(super) total_bytes: Option<u64>,
    pub(super) bytes_per_second: f64,
}

pub(super) struct BackgroundJob {
    pub(super) session_id: u64,
    pub(super) version: u64,
    pub(super) kind: JobKind,
    pub(super) stage: String,
    pub(super) progress: f32,
    pub(super) cancel: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum JobKind {
    Load,
    Recover,
    Save,
    Cartoon,
    HydrogenBonds,
    Protonation,
}

pub(super) enum JobRequest {
    Protonate {
        id: u64,
        molecule: Box<Molecule>,
        settings: astra::molecule::amoeba::protonation::Settings,
        selected: Vec<usize>,
        cancel: Arc<AtomicBool>,
    },
    HydrogenBonds {
        id: u64,
        molecule: Box<Molecule>,
        selection: String,
        settings: astra::molecule::amoeba::AnalysisSettings,
        selected: Vec<usize>,
        cancel: Arc<AtomicBool>,
    },
    ScanRecovery,
    DiscardRecovery {
        path: PathBuf,
    },
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
        molecule: Box<Molecule>,
        display: Box<DisplayState>,
        hierarchy: Box<MoleculeHierarchy>,
        secondary_structure: Vec<Vec<SecondaryStructure>>,
        cancel: Arc<AtomicBool>,
    },
}

pub(super) enum JobEvent {
    RecoveryScanned(std::result::Result<Vec<PathBuf>, String>),
    RecoveryDiscarded(std::result::Result<(), String>),
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

pub(super) enum JobOutput {
    Protonated(Box<astra::molecule::amoeba::protonation::Prepared>),
    HydrogenBonds(astra::molecule::amoeba::HydrogenBondReport),
    Loaded(Box<LoadedPayload>),
    Saved(PathBuf),
    Cartoon(PreparedCartoon),
    Cancelled,
}

pub(super) enum LoadedPayload {
    Structure {
        filename: String,
        molecule_id: String,
        molecule: Molecule,
        hierarchy: MoleculeHierarchy,
        secondary_structure: Vec<Vec<SecondaryStructure>>,
        atom_bvh: AtomBvh,
        display: Box<DisplayState>,
    },
    Scene {
        filename: String,
        path: PathBuf,
        document: Box<SceneDocument>,
        hierarchy: MoleculeHierarchy,
        secondary_structure: Vec<Vec<SecondaryStructure>>,
        atom_bvh: AtomBvh,
    },
}

#[derive(Debug)]
pub(super) enum FetchEvent {
    Progress(FetchProgress),
    Finished(Result<PathBuf, String>),
}

pub(super) fn molecule_id_from_structure(contents: &[u8], filename: &str) -> String {
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

pub(super) fn molecule_id_from_filename(filename: &str) -> String {
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

pub(super) fn is_structure_id(value: &str) -> bool {
    let value = value.trim();
    (value.len() == 4 && value.bytes().all(|byte| byte.is_ascii_alphanumeric()))
        || (value.len() == 12
            && value.to_ascii_uppercase().starts_with("PDB_")
            && value[4..].bytes().all(|byte| byte.is_ascii_alphanumeric()))
}

pub(super) fn session_label(molecule_id: Option<&str>, filename: Option<&str>) -> String {
    molecule_id
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .or_else(|| filename.map(molecule_id_from_filename))
        .unwrap_or_else(|| "Untitled".into())
}

pub(super) fn normalize_pdb_id(value: &str) -> Result<String> {
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

pub(super) fn pdb_download_directory() -> Result<PathBuf> {
    let home = env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .context("could not determine the home directory")?;
    Ok(PathBuf::from(home).join("downloads").join("pdb"))
}

#[cfg(test)]
pub(super) fn download_pdb(id: &str, directory: &Path) -> Result<PathBuf> {
    download_pdb_with_progress(id, directory, &AtomicBool::new(false), |_| {})
}

pub(super) fn download_pdb_with_progress(
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

pub(super) fn pdb_agent() -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(60)))
        .https_only(true)
        .build();
    ureq::Agent::new_with_config(config)
}

pub(super) fn download_metadata(agent: &ureq::Agent, url: &str) -> (bool, Option<u64>) {
    let Ok(response) = agent
        .head(url)
        .header("User-Agent", concat!("Astra/", env!("CARGO_PKG_VERSION")))
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

pub(super) fn download_pdb_stream(
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
        .header("User-Agent", concat!("Astra/", env!("CARGO_PKG_VERSION")))
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

pub(super) enum RangeDownloadEvent {
    Downloaded(usize),
    Finished {
        index: usize,
        contents: std::result::Result<Vec<u8>, String>,
    },
}

pub(super) fn download_pdb_ranges(
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

pub(super) fn download_pdb_range(
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
        .header("User-Agent", concat!("Astra/", env!("CARGO_PKG_VERSION")))
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

pub(super) fn pdb_download_url(id: &str) -> String {
    format!("https://files.rcsb.org/download/{id}.cif")
}

pub(super) fn read_local_file_limited(path: &Path) -> Result<Vec<u8>> {
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

pub(super) fn background_worker(
    requests: Receiver<JobRequest>,
    events: Sender<JobEvent>,
    window: Arc<Window>,
) {
    while let Ok(request) = requests.recv() {
        let (id, result) = match request {
            JobRequest::Protonate {
                id,
                molecule,
                settings,
                selected,
                cancel,
            } => {
                send_job_progress(&events, &window, id, "Preparing selected residues", 0.1);
                let result = astra::molecule::amoeba::protonation::prepare_selected(
                    &molecule, settings, &selected, &cancel,
                )
                .map(|p| JobOutput::Protonated(Box::new(p)))
                .map_err(|e| e.to_string());
                (id, result)
            }
            JobRequest::HydrogenBonds {
                id,
                molecule,
                selection,
                settings,
                selected,
                cancel,
            } => {
                send_job_progress(
                    &events,
                    &window,
                    id,
                    "AMOEBA parameterization and mutual polarization",
                    0.1,
                );
                let result = astra::molecule::amoeba::analyze(
                    &molecule, &selection, &selected, settings, &cancel,
                )
                .map(JobOutput::HydrogenBonds)
                .map_err(|error| error.to_string());
                (id, result)
            }
            JobRequest::ScanRecovery => {
                let trace = StartupTrace::new("recovery");
                trace.mark("BEGIN background recovery scan");
                let result =
                    recovery_directory().and_then(|directory| recovery::scan_files(&directory));
                trace.mark("END background recovery scan");
                let _ = events.send(JobEvent::RecoveryScanned(
                    result.map_err(|error| format!("{error:#}")),
                ));
                window.request_redraw();
                continue;
            }
            JobRequest::DiscardRecovery { path } => {
                let result = fs::remove_file(&path)
                    .with_context(|| format!("could not discard recovery {}", path.display()))
                    .map_err(|error| format!("{error:#}"));
                let _ = events.send(JobEvent::RecoveryDiscarded(result));
                window.request_redraw();
                continue;
            }
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
                molecule,
                display,
                hierarchy,
                secondary_structure,
                cancel,
            } => {
                send_job_progress(&events, &window, id, "Building ribbon geometry", 0.2);
                let result = if cancel.load(Ordering::Relaxed) {
                    Ok(JobOutput::Cancelled)
                } else {
                    let prepared = prepare_cartoon_cached(
                        &molecule,
                        &display,
                        &hierarchy,
                        &secondary_structure,
                    );
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

pub(super) fn load_in_background(
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
        let secondary_structure = assign_secondary_structure(&document.molecule, &hierarchy);
        let atom_bvh = AtomBvh::build(&document.molecule);
        return Ok(LoadedPayload::Scene {
            filename,
            path: path.to_owned(),
            document: Box::new(document),
            hierarchy,
            secondary_structure,
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
    let secondary_structure = assign_secondary_structure(&molecule, &hierarchy);
    let atom_bvh = AtomBvh::build(&molecule);
    let display = DisplayState::for_molecule(&molecule);
    let molecule_id = molecule_id_from_structure(&contents, &filename);
    Ok(LoadedPayload::Structure {
        filename,
        molecule_id,
        molecule,
        hierarchy,
        secondary_structure,
        atom_bvh,
        display: Box::new(display),
    })
}

pub(super) fn send_job_progress(
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

pub(super) fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
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

pub(super) fn recovery_directory() -> Result<PathBuf> {
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
        .join("astra/recovery");
    fs::create_dir_all(&directory)
        .with_context(|| format!("could not create {}", directory.display()))?;
    Ok(directory)
}

pub(super) fn recovery_path_for(session_id: u64) -> Result<PathBuf> {
    Ok(recovery_directory()?.join(format!(
        "astra-{}-{session_id}.recovery.mol",
        std::process::id()
    )))
}

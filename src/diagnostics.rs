use std::{fmt, io::Write, time::Instant};

/// Available even in the Windows GUI subsystem, where stderr has no console.
pub fn write_graphics_log(message: impl fmt::Display) {
    #[cfg(target_os = "windows")]
    let base = std::env::var_os("LOCALAPPDATA")
        .map(std::path::PathBuf::from)
        .map(|path| path.join("astra"));
    #[cfg(not(target_os = "windows"))]
    let base = std::env::current_dir().ok();
    let Some(base) = base else {
        return;
    };
    let directory = base.join("logs");
    if std::fs::create_dir_all(&directory).is_err() {
        return;
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(directory.join("astra-graphics.log"))
    {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let _ = writeln!(file, "[{timestamp}] {message}");
        let _ = file.flush();
    }
}

pub fn install_panic_log() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        write_graphics_log(info);
        previous(info);
    }));
}

/// Opt-in synchronous checkpoints: the last BEGIN survives a blocked driver call.
pub struct StartupTrace {
    started: Option<Instant>,
    scope: &'static str,
}

impl StartupTrace {
    pub fn new(scope: &'static str) -> Self {
        Self {
            started: (std::env::var_os("ASTRA_STARTUP_TRACE").as_deref()
                == Some(std::ffi::OsStr::new("1")))
            .then(Instant::now),
            scope,
        }
    }

    pub fn mark(&self, message: impl fmt::Display) {
        if let Some(started) = self.started {
            let mut stderr = std::io::stderr().lock();
            let _ = writeln!(
                stderr,
                "[astra:{} +{}ms] {message}",
                self.scope,
                started.elapsed().as_millis()
            );
            let _ = stderr.flush();
        }
    }
}

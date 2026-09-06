use std::{fmt, io::Write, time::Instant};

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

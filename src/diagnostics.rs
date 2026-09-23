//! Application logging and startup tracing.
//!
//! Records go through the `log` facade to a size-rotated file in the platform log directory
//! (see [`crate::paths::log_dir`]), which also works in the Windows GUI subsystem where stderr
//! has no console. `ASTRA_LOG` selects the level (`error`, `warn`, `info`, `debug`, `trace`);
//! `ASTRA_LOG_STDERR=1` mirrors records to stderr.

use std::{
    fmt,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use log::{Level, LevelFilter, Log, Metadata, Record};

const LOG_FILE_NAME: &str = "astra.log";
const MAX_LOG_BYTES: u64 = 4 * 1024 * 1024;
const ROTATED_FILES: usize = 3;

struct FileLogger {
    level: LevelFilter,
    stderr: bool,
    sink: Mutex<Option<LogFile>>,
}

struct LogFile {
    path: PathBuf,
    file: File,
    written: u64,
}

impl LogFile {
    fn open(directory: &Path) -> std::io::Result<Self> {
        fs::create_dir_all(directory)?;
        let path = directory.join(LOG_FILE_NAME);
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let written = file.metadata().map(|metadata| metadata.len()).unwrap_or(0);
        Ok(Self {
            path,
            file,
            written,
        })
    }

    fn write_line(&mut self, line: &str) {
        if self.written + line.len() as u64 > MAX_LOG_BYTES {
            self.rotate();
        }
        if self.file.write_all(line.as_bytes()).is_ok() {
            self.written += line.len() as u64;
        }
    }

    /// `astra.log` becomes `astra.log.1`, older files shift up and the oldest is removed.
    fn rotate(&mut self) {
        let _ = self.file.flush();
        for index in (1..ROTATED_FILES).rev() {
            let _ = fs::rename(rotated(&self.path, index), rotated(&self.path, index + 1));
        }
        let _ = fs::rename(&self.path, rotated(&self.path, 1));
        if let Ok(file) = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)
        {
            self.file = file;
            self.written = 0;
        }
    }
}

fn rotated(path: &Path, index: usize) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(format!(".{index}"));
    PathBuf::from(name)
}

impl Log for FileLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= self.level
            && (metadata.target().starts_with("astra") || metadata.level() <= Level::Warn)
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = format!(
            "{} {:<5} [{}] {}\n",
            utc_timestamp(SystemTime::now()),
            record.level(),
            record.target(),
            record.args()
        );
        if self.stderr {
            let _ = std::io::stderr().lock().write_all(line.as_bytes());
        }
        if let Ok(mut sink) = self.sink.lock()
            && let Some(sink) = sink.as_mut()
        {
            sink.write_line(&line);
            if record.level() <= Level::Warn {
                let _ = sink.file.flush();
            }
        }
    }

    fn flush(&self) {
        if let Ok(mut sink) = self.sink.lock()
            && let Some(sink) = sink.as_mut()
        {
            let _ = sink.file.flush();
        }
    }
}

static LOGGER: OnceLock<FileLogger> = OnceLock::new();

/// Installs the file logger and the panic hook. Returns the log file path when one could be
/// opened; logging failures never prevent the application from starting.
pub fn init() -> Option<PathBuf> {
    let level = std::env::var("ASTRA_LOG")
        .ok()
        .and_then(|value| value.parse::<LevelFilter>().ok())
        .unwrap_or(LevelFilter::Info);
    let stderr = std::env::var_os("ASTRA_LOG_STDERR").as_deref() == Some(std::ffi::OsStr::new("1"));
    let sink = crate::paths::log_dir().and_then(|directory| LogFile::open(&directory).ok());
    let path = sink.as_ref().map(|sink| sink.path.clone());
    let logger = LOGGER.get_or_init(|| FileLogger {
        level,
        stderr,
        sink: Mutex::new(sink),
    });
    if log::set_logger(logger).is_ok() {
        log::set_max_level(level);
    }
    install_panic_log();
    log::info!(
        "Astra {} starting on {} {}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    path
}

/// Graphics startup and device diagnostics, recorded at the default level.
pub fn write_graphics_log(message: impl fmt::Display) {
    log::info!(target: "astra::graphics", "{message}");
}

fn install_panic_log() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        log::error!(target: "astra::panic", "{info}\n{backtrace}");
        log::logger().flush();
        previous(info);
    }));
}

/// RFC 3339 UTC timestamp with millisecond precision, without a date-time dependency.
fn utc_timestamp(time: SystemTime) -> String {
    let elapsed = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let seconds = elapsed.as_secs();
    let (year, month, day) = civil_from_days((seconds / 86_400) as i64);
    let second_of_day = seconds % 86_400;
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        second_of_day / 3600,
        second_of_day / 60 % 60,
        second_of_day % 60,
        elapsed.subsec_millis()
    )
}

/// Howard Hinnant's days-to-civil conversion for the proleptic Gregorian calendar.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn formats_utc_timestamps() {
        assert_eq!(utc_timestamp(UNIX_EPOCH), "1970-01-01T00:00:00.000Z");
        let leap_day = UNIX_EPOCH + Duration::from_millis(951_827_696_789);
        assert_eq!(utc_timestamp(leap_day), "2000-02-29T12:34:56.789Z");
        let later = UNIX_EPOCH + Duration::from_secs(1_790_208_000);
        assert_eq!(utc_timestamp(later), "2026-09-24T00:00:00.000Z");
    }

    #[test]
    fn rotates_when_the_size_limit_is_reached() {
        let directory = std::env::temp_dir().join(format!("astra-log-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        let mut sink = LogFile::open(&directory).unwrap();
        sink.written = MAX_LOG_BYTES;
        sink.write_line("after rotation\n");
        sink.file.flush().unwrap();
        assert_eq!(
            fs::read_to_string(directory.join(LOG_FILE_NAME)).unwrap(),
            "after rotation\n"
        );
        assert!(directory.join("astra.log.1").exists());
        fs::remove_dir_all(&directory).unwrap();
    }
}

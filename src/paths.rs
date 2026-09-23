//! Per-user directories following each platform's conventions.
//!
//! Every function honors an `ASTRA_*` override so packaged builds, tests and portable installs
//! can redirect state without touching the user's real profile.

use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
};

fn var(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn home() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        var("USERPROFILE").or_else(|| var("HOME"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        var("HOME")
    }
}

/// Persistent application state: recovery files and similar data that must survive restarts.
pub fn state_dir() -> Option<PathBuf> {
    if let Some(path) = var("ASTRA_STATE_DIR") {
        return Some(path);
    }
    #[cfg(target_os = "windows")]
    let base = var("LOCALAPPDATA");
    #[cfg(target_os = "macos")]
    let base = home().map(|home| home.join("Library/Application Support"));
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let base = var("XDG_STATE_HOME").or_else(|| home().map(|home| home.join(".local/state")));
    // Lowercase on every platform: earlier releases stored recovery files here.
    base.map(|base| base.join("astra"))
}

/// Diagnostic logs.
pub fn log_dir() -> Option<PathBuf> {
    if let Some(path) = var("ASTRA_LOG_DIR") {
        return Some(path);
    }
    #[cfg(target_os = "macos")]
    {
        home().map(|home| home.join("Library/Logs/Astra"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        state_dir().map(|state| state.join("logs"))
    }
}

/// The user's download folder. Structures fetched from the PDB are stored in its `pdb`
/// subdirectory, where they remain available after Astra closes.
pub fn downloads_dir() -> Option<PathBuf> {
    if let Some(path) = var("ASTRA_DOWNLOAD_DIR") {
        return Some(path);
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    if let Some(path) = var("XDG_DOWNLOAD_DIR").or_else(xdg_user_download_dir) {
        return Some(path);
    }
    home().map(|home| home.join("Downloads"))
}

pub fn pdb_download_dir() -> Option<PathBuf> {
    downloads_dir().map(|path| path.join("pdb"))
}

/// Reads `XDG_DOWNLOAD_DIR` from `user-dirs.dirs`, the file maintained by xdg-user-dirs.
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn xdg_user_download_dir() -> Option<PathBuf> {
    let home = home()?;
    let config = var("XDG_CONFIG_HOME").unwrap_or_else(|| home.join(".config"));
    let text = std::fs::read_to_string(config.join("user-dirs.dirs")).ok()?;
    parse_user_dirs(&text, "XDG_DOWNLOAD_DIR", &home)
}

#[cfg_attr(any(target_os = "windows", target_os = "macos"), allow(dead_code))]
fn parse_user_dirs(text: &str, key: &str, home: &Path) -> Option<PathBuf> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| line.split_once('='))
        .find(|(name, _)| name.trim() == key)
        .map(|(_, value)| value.trim().trim_matches('"'))
        .filter(|value| !value.is_empty())
        .map(|value| {
            if let Some(rest) = value.strip_prefix("$HOME") {
                let mut path = OsString::from(home.as_os_str());
                path.push(rest);
                PathBuf::from(path)
            } else {
                PathBuf::from(value)
            }
        })
}

/// A path shortened with `~` for display when it lies under the home directory.
pub fn display_path(path: &Path) -> String {
    if let Some(home) = home()
        && let Ok(relative) = path.strip_prefix(&home)
    {
        return Path::new("~").join(relative).display().to_string();
    }
    path.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_xdg_user_dirs_with_home_expansion() {
        let text =
            "# comment\nXDG_DESKTOP_DIR=\"$HOME/Desktop\"\nXDG_DOWNLOAD_DIR=\"$HOME/Загрузки\"\n";
        assert_eq!(
            parse_user_dirs(text, "XDG_DOWNLOAD_DIR", Path::new("/home/a")),
            Some(PathBuf::from("/home/a/Загрузки"))
        );
        assert_eq!(
            parse_user_dirs(
                "XDG_DOWNLOAD_DIR=\"/data/dl\"",
                "XDG_DOWNLOAD_DIR",
                Path::new("/h")
            ),
            Some(PathBuf::from("/data/dl"))
        );
        assert_eq!(
            parse_user_dirs("", "XDG_DOWNLOAD_DIR", Path::new("/h")),
            None
        );
    }
}

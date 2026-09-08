#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod app;
mod ui;

use std::path::PathBuf;

use anyhow::Result;

fn main() -> Result<()> {
    astra::diagnostics::install_panic_log();
    let result = app::run(std::env::args_os().nth(1).map(PathBuf::from));
    if let Err(error) = &result {
        astra::diagnostics::write_graphics_log(format_args!("Application error: {error:#}"));
    }
    result
}

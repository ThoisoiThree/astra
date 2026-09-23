#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod app;
mod ui;

use std::path::PathBuf;

use anyhow::Result;

fn main() -> Result<()> {
    astra::diagnostics::init();
    let result = app::run(std::env::args_os().nth(1).map(PathBuf::from));
    if let Err(error) = &result {
        log::error!("Application error: {error:#}");
        log::logger().flush();
    }
    result
}

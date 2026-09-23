#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod app;
mod cli;
mod ui;

use anyhow::Result;

fn main() -> Result<()> {
    astra::diagnostics::init();
    let options = match cli::parse(std::env::args_os().skip(1)) {
        Ok(cli::Parsed::Run(options)) => options,
        Ok(cli::Parsed::Exit(message)) => {
            println!("{message}");
            return Ok(());
        }
        Err(error) => {
            eprintln!("astra: {error:#}");
            std::process::exit(2);
        }
    };
    let result = app::run(options);
    if let Err(error) = &result {
        log::error!("Application error: {error:#}");
        log::logger().flush();
    }
    result
}

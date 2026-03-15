use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum CliOutputFormat {
    #[default]
    Text,
    Json,
}

#[derive(Parser, Debug, Clone, Default)]
#[command(name = "handy", about = "Handy - Speech to Text")]
pub struct CliArgs {
    /// Transcribe a local WAV file (headless mode, no UI/tray)
    #[arg(long)]
    pub transcribe_file: Option<PathBuf>,

    /// Output format for --transcribe-file mode
    #[arg(long, value_enum, default_value_t = CliOutputFormat::Text)]
    pub output_format: CliOutputFormat,

    /// Override selected model for --transcribe-file mode (e.g. sense-voice-small)
    #[arg(long)]
    pub model: Option<String>,

    /// Override language for --transcribe-file mode (e.g. auto, zh, zh-Hans, en)
    #[arg(long)]
    pub language: Option<String>,

    /// Start with the main window hidden
    #[arg(long)]
    pub start_hidden: bool,

    /// Disable the system tray icon
    #[arg(long)]
    pub no_tray: bool,

    /// Toggle transcription on/off (sent to running instance)
    #[arg(long)]
    pub toggle_transcription: bool,

    /// Toggle transcription with post-processing on/off (sent to running instance)
    #[arg(long)]
    pub toggle_post_process: bool,

    /// Cancel the current operation (sent to running instance)
    #[arg(long)]
    pub cancel: bool,

    /// Enable debug mode with verbose logging
    #[arg(long)]
    pub debug: bool,
}

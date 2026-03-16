use crate::cli::{CliArgs, CliOutputFormat};
use crate::tauri_context;
use crate::managers::model::ModelManager;
use crate::managers::transcription::TranscriptionManager;
use crate::settings::{get_settings, write_settings, AppSettings};
use anyhow::{anyhow, Result};
use serde::Serialize;
use std::path::Path;
use std::sync::Arc;
use tauri::AppHandle;

#[derive(Serialize)]
struct JsonTranscriptionOutput {
    text: String,
    model: String,
    language: String,
    file: String,
}

#[derive(Serialize)]
struct JsonErrorOutput {
    error: String,
}

#[derive(Serialize)]
struct JsonModelOutput {
    id: String,
    name: String,
    downloaded: bool,
    selected: bool,
}

struct SettingsRestoreGuard {
    app_handle: AppHandle,
    original_settings: AppSettings,
    should_restore: bool,
}

impl SettingsRestoreGuard {
    fn new(app_handle: AppHandle, original_settings: AppSettings, should_restore: bool) -> Self {
        Self {
            app_handle,
            original_settings,
            should_restore,
        }
    }
}

impl Drop for SettingsRestoreGuard {
    fn drop(&mut self) {
        if self.should_restore {
            write_settings(&self.app_handle, self.original_settings.clone());
        }
    }
}

pub fn run(cli_args: CliArgs) {
    let cli_args_for_setup = cli_args.clone();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::default().build())
        .setup(move |app| {
            let exit_code = if cli_args_for_setup.list_models {
                match list_models(&app.handle(), cli_args_for_setup.output_format) {
                    Ok(()) => 0,
                    Err(err) => {
                        match cli_args_for_setup.output_format {
                            CliOutputFormat::Text => eprintln!("Error: {}", err),
                            CliOutputFormat::Json => {
                                let output = JsonErrorOutput {
                                    error: err.to_string(),
                                };
                                eprintln!(
                                    "{}",
                                    serde_json::to_string(&output)
                                        .expect("Failed to serialize error JSON output")
                                );
                            }
                        }
                        1
                    }
                }
            } else {
                match transcribe_once(&app.handle(), &cli_args_for_setup) {
                    Ok(result) => {
                        match cli_args_for_setup.output_format {
                            CliOutputFormat::Text => println!("{}", result.text),
                            CliOutputFormat::Json => {
                                let output = JsonTranscriptionOutput {
                                    text: result.text,
                                    model: result.model,
                                    language: result.language,
                                    file: result.file,
                                };
                                println!(
                                    "{}",
                                    serde_json::to_string(&output)
                                        .expect("Failed to serialize transcription JSON output")
                                );
                            }
                        }
                        0
                    }
                    Err(err) => {
                        match cli_args_for_setup.output_format {
                            CliOutputFormat::Text => eprintln!("Error: {}", err),
                            CliOutputFormat::Json => {
                                let output = JsonErrorOutput {
                                    error: err.to_string(),
                                };
                                eprintln!(
                                    "{}",
                                    serde_json::to_string(&output)
                                        .expect("Failed to serialize error JSON output")
                                );
                            }
                        }
                        1
                    }
                }
            };

            std::process::exit(exit_code);
        })
        .build(tauri_context())
        .expect("error while building headless CLI application");

    app.run(|_app_handle, _event| {});
}

struct OneShotTranscriptionResult {
    text: String,
    model: String,
    language: String,
    file: String,
}

fn list_models(app_handle: &AppHandle, output_format: CliOutputFormat) -> Result<()> {
    let model_manager = ModelManager::new(app_handle)?;
    let selected_model = get_settings(app_handle).selected_model;

    let mut models = model_manager.get_available_models();
    models.sort_by(|a, b| a.id.cmp(&b.id));

    match output_format {
        CliOutputFormat::Text => {
            for model in models.iter().filter(|m| m.is_downloaded) {
                let selected_marker = if model.id == selected_model { " *selected" } else { "" };
                println!("{}\t{}{}", model.id, model.name, selected_marker);
            }
        }
        CliOutputFormat::Json => {
            let output: Vec<JsonModelOutput> = models
                .into_iter()
                .filter(|m| m.is_downloaded)
                .map(|m| JsonModelOutput {
                    selected: m.id == selected_model,
                    id: m.id,
                    name: m.name,
                    downloaded: m.is_downloaded,
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string(&output)
                    .expect("Failed to serialize model list JSON output")
            );
        }
    }

    Ok(())
}

fn transcribe_once(app_handle: &AppHandle, cli_args: &CliArgs) -> Result<OneShotTranscriptionResult> {
    let audio_path = cli_args
        .transcribe_file
        .as_deref()
        .ok_or_else(|| anyhow!("Missing --transcribe-file path"))?;
    ensure_file_exists(audio_path)?;

    let original_settings = get_settings(app_handle);
    let mut effective_settings = original_settings.clone();
    if let Some(model) = &cli_args.model {
        effective_settings.selected_model = model.clone();
    }
    if let Some(language) = &cli_args.language {
        effective_settings.selected_language = language.clone();
    }

    let should_restore = cli_args.model.is_some() || cli_args.language.is_some();
    if should_restore {
        write_settings(app_handle, effective_settings.clone());
    }
    let _settings_restore =
        SettingsRestoreGuard::new(app_handle.clone(), original_settings, should_restore);

    let model_manager = Arc::new(ModelManager::new(app_handle)?);
    if effective_settings.selected_model.trim().is_empty() {
        effective_settings.selected_model = get_settings(app_handle).selected_model;
    }

    let selected_model = effective_settings.selected_model.trim().to_string();
    if selected_model.is_empty() {
        return Err(anyhow!(
            "No model selected. Set one in Handy settings or pass --model <model-id>."
        ));
    }

    let model_info = model_manager
        .get_model_info(&selected_model)
        .ok_or_else(|| anyhow!("Model not found: {}", selected_model))?;

    if !model_info.is_downloaded {
        return Err(anyhow!(
            "Model '{}' is not downloaded. Download it in Handy first.",
            selected_model
        ));
    }

    let transcription_manager = TranscriptionManager::new(app_handle, model_manager)?;
    transcription_manager.load_model(&selected_model)?;

    let audio = transcribe_rs::audio::read_wav_samples(audio_path).map_err(|e| {
        anyhow!(
            "Failed to read audio file '{}': {}. Expected 16kHz, 16-bit, mono PCM WAV.",
            audio_path.display(),
            e
        )
    })?;

    let text = transcription_manager.transcribe(audio)?;
    std::mem::forget(transcription_manager);

    Ok(OneShotTranscriptionResult {
        text,
        model: selected_model,
        language: effective_settings.selected_language,
        file: audio_path.to_string_lossy().to_string(),
    })
}

fn ensure_file_exists(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(anyhow!("Audio file not found: {}", path.display()));
    }
    if !path.is_file() {
        return Err(anyhow!("Path is not a file: {}", path.display()));
    }
    Ok(())
}

use anyhow::{anyhow, Result};
use std::env;
use std::path::{Path, PathBuf};
use transcribe_rs::{
    audio::read_wav_samples,
    engines::{
        gigaam::GigaAMEngine,
        moonshine::{
            ModelVariant, MoonshineEngine, MoonshineModelParams, MoonshineStreamingEngine,
            StreamingModelParams,
        },
        parakeet::{
            ParakeetEngine, ParakeetInferenceParams, ParakeetModelParams, TimestampGranularity,
        },
        sense_voice::{
            Language as SenseVoiceLanguage, SenseVoiceEngine, SenseVoiceInferenceParams,
            SenseVoiceModelParams,
        },
        whisper::{WhisperEngine, WhisperInferenceParams},
    },
    TranscriptionEngine,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EngineType {
    Whisper,
    Parakeet,
    Moonshine,
    MoonshineStreaming,
    SenseVoice,
    GigaAM,
}

#[derive(Clone, Copy, Debug)]
struct ModelSpec {
    id: &'static str,
    filename: &'static str,
    is_directory: bool,
    engine_type: EngineType,
}

const MODEL_SPECS: &[ModelSpec] = &[
    ModelSpec { id: "small", filename: "ggml-small.bin", is_directory: false, engine_type: EngineType::Whisper },
    ModelSpec { id: "medium", filename: "whisper-medium-q4_1.bin", is_directory: false, engine_type: EngineType::Whisper },
    ModelSpec { id: "turbo", filename: "ggml-large-v3-turbo.bin", is_directory: false, engine_type: EngineType::Whisper },
    ModelSpec { id: "large", filename: "ggml-large-v3-q5_0.bin", is_directory: false, engine_type: EngineType::Whisper },
    ModelSpec { id: "breeze-asr", filename: "breeze-asr-q5_k.bin", is_directory: false, engine_type: EngineType::Whisper },
    ModelSpec { id: "parakeet-tdt-0.6b-v2", filename: "parakeet-tdt-0.6b-v2-int8", is_directory: true, engine_type: EngineType::Parakeet },
    ModelSpec { id: "parakeet-tdt-0.6b-v3", filename: "parakeet-tdt-0.6b-v3-int8", is_directory: true, engine_type: EngineType::Parakeet },
    ModelSpec { id: "moonshine-base", filename: "moonshine-base", is_directory: true, engine_type: EngineType::Moonshine },
    ModelSpec { id: "moonshine-tiny-streaming-en", filename: "moonshine-tiny-streaming-en", is_directory: true, engine_type: EngineType::MoonshineStreaming },
    ModelSpec { id: "moonshine-small-streaming-en", filename: "moonshine-small-streaming-en", is_directory: true, engine_type: EngineType::MoonshineStreaming },
    ModelSpec { id: "moonshine-medium-streaming-en", filename: "moonshine-medium-streaming-en", is_directory: true, engine_type: EngineType::MoonshineStreaming },
    ModelSpec { id: "sense-voice-int8", filename: "sense-voice-int8", is_directory: true, engine_type: EngineType::SenseVoice },
    ModelSpec { id: "gigaam-v3-e2e-ctc", filename: "giga-am-v3.int8.onnx", is_directory: false, engine_type: EngineType::GigaAM },
];

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranscriptionOutput {
    pub text: String,
    pub model: String,
    pub language: String,
    pub file: String,
}

pub struct LoadedTranscriptionModel {
    model_id: String,
    engine_type: EngineType,
    engine: LoadedEngine,
}

enum LoadedEngine {
    Whisper(WhisperEngine),
    Parakeet(ParakeetEngine),
    Moonshine(MoonshineEngine),
    MoonshineStreaming(MoonshineStreamingEngine),
    SenseVoice(SenseVoiceEngine),
    GigaAM(GigaAMEngine),
}

fn portable_data_dir() -> Option<PathBuf> {
    let exe_path = env::current_exe().ok()?;
    let exe_dir = exe_path.parent()?;
    if exe_dir.join("portable").exists() {
        Some(exe_dir.join("Data"))
    } else {
        None
    }
}

fn default_models_dir() -> Result<PathBuf> {
    if let Ok(path) = env::var("HANDY_MODELS_DIR") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }

    if let Some(portable) = portable_data_dir() {
        return Ok(portable.join("models"));
    }

    #[cfg(target_os = "macos")]
    {
        let home = env::var("HOME").map_err(|_| anyhow!("HOME is not set"))?;
        return Ok(PathBuf::from(home).join("Library/Application Support/com.pais.handy/models"));
    }

    #[cfg(target_os = "windows")]
    {
        let appdata = env::var("APPDATA").map_err(|_| anyhow!("APPDATA is not set"))?;
        return Ok(PathBuf::from(appdata).join("com.pais.handy/models"));
    }

    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        let home = env::var("HOME").map_err(|_| anyhow!("HOME is not set"))?;
        return Ok(PathBuf::from(home).join(".local/share/com.pais.handy/models"));
    }
}

fn custom_model_spec(model_id: &str, models_dir: &Path) -> Option<ModelSpec> {
    let path = models_dir.join(format!("{model_id}.bin"));
    if path.is_file() {
        Some(ModelSpec {
            id: Box::leak(model_id.to_string().into_boxed_str()),
            filename: Box::leak(format!("{model_id}.bin").into_boxed_str()),
            is_directory: false,
            engine_type: EngineType::Whisper,
        })
    } else {
        None
    }
}

fn resolve_model_spec(model_id: &str, models_dir: &Path) -> Option<ModelSpec> {
    MODEL_SPECS
        .iter()
        .find(|spec| spec.id == model_id)
        .copied()
        .or_else(|| custom_model_spec(model_id, models_dir))
}

fn model_path(spec: ModelSpec, models_dir: &Path) -> PathBuf {
    models_dir.join(spec.filename)
}

fn model_exists(spec: ModelSpec, models_dir: &Path) -> bool {
    let path = model_path(spec, models_dir);
    if spec.is_directory { path.is_dir() } else { path.is_file() }
}

fn resolve_model(model_id: &str) -> Result<(ModelSpec, PathBuf)> {
    let models_dir = default_models_dir()?;
    let spec = resolve_model_spec(model_id, &models_dir)
        .ok_or_else(|| anyhow!("Model not found: {model_id}"))?;
    let resolved_model_path = model_path(spec, &models_dir);

    if !model_exists(spec, &models_dir) {
        return Err(anyhow!("Model '{}' is not downloaded. Download it in Handy first.", model_id));
    }

    Ok((spec, resolved_model_path))
}

pub fn list_downloaded_models() -> Result<Vec<String>> {
    let models_dir = default_models_dir()?;
    let mut result: Vec<String> = MODEL_SPECS
        .iter()
        .copied()
        .filter(|spec| model_exists(*spec, &models_dir))
        .map(|spec| spec.id.to_string())
        .collect();

    if models_dir.is_dir() {
        for entry in std::fs::read_dir(&models_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else { continue; };
            if !name.ends_with(".bin") { continue; }
            let Some(model_id) = name.strip_suffix(".bin") else { continue; };
            if MODEL_SPECS.iter().any(|spec| spec.id == model_id) { continue; }
            result.push(model_id.to_string());
        }
    }

    result.sort();
    Ok(result)
}

fn whisper_language(language: &str) -> Option<String> {
    match language {
        "auto" => None,
        "zh-Hans" | "zh-Hant" => Some("zh".to_string()),
        other => Some(other.to_string()),
    }
}

fn whisper_initial_prompt(language: &str) -> Option<String> {
    if matches!(language, "zh" | "zh-Hans" | "zh-Hant" | "auto") {
        Some("The transcript is mainly in Chinese, but may include English words, product names, and technical terms. Use natural Chinese punctuation when appropriate, including ， 。 ？ ！ and quotation marks when needed. Preserve the original meaning and wording as much as possible.".to_string())
    } else {
        None
    }
}

fn engine_err<E: std::fmt::Display>(err: E) -> anyhow::Error {
    anyhow!(err.to_string())
}

impl LoadedTranscriptionModel {
    pub fn load(model_id: &str) -> Result<Self> {
        let (spec, resolved_model_path) = resolve_model(model_id)?;
        let engine = match spec.engine_type {
            EngineType::Whisper => {
                let mut engine = WhisperEngine::new();
                engine.load_model(&resolved_model_path).map_err(engine_err)?;
                LoadedEngine::Whisper(engine)
            }
            EngineType::Parakeet => {
                let mut engine = ParakeetEngine::new();
                engine
                    .load_model_with_params(&resolved_model_path, ParakeetModelParams::int8())
                    .map_err(engine_err)?;
                LoadedEngine::Parakeet(engine)
            }
            EngineType::Moonshine => {
                let mut engine = MoonshineEngine::new();
                engine
                    .load_model_with_params(
                        &resolved_model_path,
                        MoonshineModelParams::variant(ModelVariant::Base),
                    )
                    .map_err(engine_err)?;
                LoadedEngine::Moonshine(engine)
            }
            EngineType::MoonshineStreaming => {
                let mut engine = MoonshineStreamingEngine::new();
                engine
                    .load_model_with_params(&resolved_model_path, StreamingModelParams::default())
                    .map_err(engine_err)?;
                LoadedEngine::MoonshineStreaming(engine)
            }
            EngineType::SenseVoice => {
                let mut engine = SenseVoiceEngine::new();
                engine
                    .load_model_with_params(&resolved_model_path, SenseVoiceModelParams::int8())
                    .map_err(engine_err)?;
                LoadedEngine::SenseVoice(engine)
            }
            EngineType::GigaAM => {
                let mut engine = GigaAMEngine::new();
                engine.load_model(&resolved_model_path).map_err(engine_err)?;
                LoadedEngine::GigaAM(engine)
            }
        };

        Ok(Self {
            model_id: model_id.to_string(),
            engine_type: spec.engine_type,
            engine,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn transcribe_wav_file(&mut self, language: &str, wav_path: &Path) -> Result<TranscriptionOutput> {
        if !wav_path.exists() {
            return Err(anyhow!("Audio file not found: {}", wav_path.display()));
        }
        if !wav_path.is_file() {
            return Err(anyhow!("Path is not a file: {}", wav_path.display()));
        }

        let audio = read_wav_samples(wav_path).map_err(|e| {
            anyhow!(
                "Failed to read audio file '{}': {}. Expected 16kHz, 16-bit, mono PCM WAV.",
                wav_path.display(),
                e
            )
        })?;

        let text = match (&self.engine_type, &mut self.engine) {
            (EngineType::Whisper, LoadedEngine::Whisper(engine)) => {
                let params = WhisperInferenceParams {
                    language: whisper_language(language),
                    initial_prompt: whisper_initial_prompt(language),
                    ..Default::default()
                };
                engine.transcribe_samples(audio, Some(params)).map_err(engine_err)?.text
            }
            (EngineType::Parakeet, LoadedEngine::Parakeet(engine)) => engine
                .transcribe_samples(
                    audio,
                    Some(ParakeetInferenceParams {
                        timestamp_granularity: TimestampGranularity::Segment,
                        ..Default::default()
                    }),
                )
                .map_err(engine_err)?
                .text,
            (EngineType::Moonshine, LoadedEngine::Moonshine(engine)) => {
                engine.transcribe_samples(audio, None).map_err(engine_err)?.text
            }
            (EngineType::MoonshineStreaming, LoadedEngine::MoonshineStreaming(engine)) => {
                engine.transcribe_samples(audio, None).map_err(engine_err)?.text
            }
            (EngineType::SenseVoice, LoadedEngine::SenseVoice(engine)) => {
                let sense_voice_language = match language {
                    "zh" | "zh-Hans" | "zh-Hant" => SenseVoiceLanguage::Chinese,
                    "en" => SenseVoiceLanguage::English,
                    "ja" => SenseVoiceLanguage::Japanese,
                    "ko" => SenseVoiceLanguage::Korean,
                    "yue" => SenseVoiceLanguage::Cantonese,
                    _ => SenseVoiceLanguage::Auto,
                };
                engine
                    .transcribe_samples(
                        audio,
                        Some(SenseVoiceInferenceParams {
                            language: sense_voice_language,
                            use_itn: true,
                        }),
                    )
                    .map_err(engine_err)?
                    .text
            }
            (EngineType::GigaAM, LoadedEngine::GigaAM(engine)) => {
                engine.transcribe_samples(audio, None).map_err(engine_err)?.text
            }
            _ => return Err(anyhow!("Loaded engine type mismatch for model '{}'", self.model_id)),
        };

        Ok(TranscriptionOutput {
            text,
            model: self.model_id.clone(),
            language: language.to_string(),
            file: wav_path.to_string_lossy().to_string(),
        })
    }

    pub fn unload(&mut self) {
        match &mut self.engine {
            LoadedEngine::Whisper(engine) => engine.unload_model(),
            LoadedEngine::Parakeet(engine) => engine.unload_model(),
            LoadedEngine::Moonshine(engine) => engine.unload_model(),
            LoadedEngine::MoonshineStreaming(engine) => engine.unload_model(),
            LoadedEngine::SenseVoice(engine) => engine.unload_model(),
            LoadedEngine::GigaAM(engine) => engine.unload_model(),
        }
    }
}

impl Drop for LoadedTranscriptionModel {
    fn drop(&mut self) {
        self.unload();
    }
}

pub fn transcribe_wav_file(model: &str, language: &str, wav_path: &Path) -> Result<TranscriptionOutput> {
    LoadedTranscriptionModel::load(model)?.transcribe_wav_file(language, wav_path)
}

import React from "react";
import { useTranslation } from "react-i18next";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { LanguageSelector } from "../LanguageSelector";
import { TranslateToEnglish } from "../TranslateToEnglish";
import { useModelStore } from "../../../stores/modelStore";
import { useSettingsStore } from "../../../stores/settingsStore";
import type { ModelInfo } from "@/bindings";

export const ModelSettingsCard: React.FC = () => {
  const { t } = useTranslation();
  const { currentModel, models } = useModelStore();
  const clientModeEnabled = useSettingsStore((state) =>
    state.settings?.client_mode_enabled ?? false,
  );

  const currentModelInfo = models.find((m: ModelInfo) => m.id === currentModel);

  const supportsLanguageSelection =
    clientModeEnabled ||
    currentModelInfo?.engine_type === "Whisper" ||
    currentModelInfo?.engine_type === "SenseVoice";
  const supportsTranslation =
    clientModeEnabled || (currentModelInfo?.supports_translation ?? false);
  const hasAnySettings = supportsLanguageSelection || supportsTranslation;

  // In client mode, keep language/translation settings even without a local model.
  if (!clientModeEnabled && (!currentModel || !currentModelInfo || !hasAnySettings)) {
    return null;
  }

  return (
    <SettingsGroup
      title={
        clientModeEnabled
          ? "Transcription Settings"
          : t("settings.modelSettings.title", {
              model: currentModelInfo?.name || "",
            })
      }
    >
      {supportsLanguageSelection && (
        <LanguageSelector
          descriptionMode="tooltip"
          grouped={true}
          supportedLanguages={currentModelInfo?.supported_languages}
        />
      )}
      {supportsTranslation && (
        <TranslateToEnglish descriptionMode="tooltip" grouped={true} />
      )}
    </SettingsGroup>
  );
};

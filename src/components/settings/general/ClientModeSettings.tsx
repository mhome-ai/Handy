import React from "react";
import { useSettings } from "../../../hooks/useSettings";
import { ToggleSwitch } from "../../ui/ToggleSwitch";
import { SettingContainer } from "../../ui/SettingContainer";
import { Input } from "../../ui/Input";
import { ResetButton } from "../../ui/ResetButton";

export const ClientModeSettings: React.FC = () => {
  const { getSetting, updateSetting, resetSetting, isUpdating } = useSettings();

  const enabled = getSetting("client_mode_enabled") ?? false;
  const baseUrl = getSetting("client_mode_base_url") || "";

  return (
    <>
      <ToggleSwitch
        label="Client Mode"
        description="Record locally, then send the audio to a remote gateway for transcription instead of using a local model."
        checked={enabled}
        onChange={(checked) => updateSetting("client_mode_enabled", checked)}
        isUpdating={isUpdating("client_mode_enabled")}
        descriptionMode="tooltip"
        grouped={true}
      />
      <SettingContainer
        title="Remote Transcription URL"
        description="Base URL of the remote gateway, for example http://192.168.86.21:9999"
        descriptionMode="tooltip"
        grouped={true}
        disabled={!enabled}
      >
        <div className="flex items-center gap-2">
          <Input
            value={baseUrl}
            disabled={!enabled || isUpdating("client_mode_base_url")}
            onChange={(e) => updateSetting("client_mode_base_url", e.target.value)}
            placeholder="http://192.168.86.21:9999"
            className="min-w-[280px]"
          />
          <ResetButton
            onClick={() => resetSetting("client_mode_base_url")}
            disabled={!enabled || isUpdating("client_mode_base_url")}
          />
        </div>
      </SettingContainer>
    </>
  );
};

"""Provider/model selection and the reasoning-effort persistence quirk.

Invariant: when the provider rejects reasoning_effort, the rejection is the
source of truth. The agent retries without the parameter and
_persist_adjusted_thinking writes the correction to the profile. There is no
per-model table predicting this ahead of time; do not add one.
"""
from pathlib import Path

import config
from cli_i18n import t
from cli_presentation import _short_context


class ProvidersMixin:
    def _provider_from_profile(self, item):
        if not item or item.get("provider", "ollama") == "ollama":
            return {"provider": "ollama"}
        secret_path = (Path(self.config_file).with_name("secrets.json")
                       if self.config_file else None)
        provider = {
            "provider": "openai_compatible",
            "provider_name": item.get("provider_name") or "API",
            "base_url": item.get("base_url"),
            "api_key": config.load_secret(item.get("credential"), secret_path),
        }
        # Optional {"cmd": [...], "health_url": "..."}: lets isaacli start and
        # stop this server the way it already does for Ollama, instead of only
        # checking whether the endpoint happens to be up already.
        autostart = item.get("autostart")
        if autostart:
            provider["autostart"] = autostart
        return provider

    def _persist_adjusted_thinking(self):
        """Write thinking=None to the active profile after the provider rejected
        the configured reasoning effort, so the error (and the extra round trip
        it costs) does not repeat in every future conversation. Returns False
        when there is nowhere to persist it. The caller has to tell the user
        that, not swallow the failure."""
        try:
            data = config.load(self.config_file)
        except ValueError as e:
            self._log("error", error=f"thinking_adjusted: unreadable configuration ({e})")
            return False
        name, item = config.profile(data)
        if not item or item.get("model") != self.model:
            self._log("error", error="thinking_adjusted: no saved profile matches "
                      f"the active model ({self.model})")
            return False
        item["thinking"] = None
        data["profiles"][name] = item
        config.save(data, self.config_file)
        return True

    def select_model(self):
        import setup_ollama

        code = setup_ollama.run_model_selector(config_file=self.config_file)
        if code != 0:
            message = (t("cli.model.selection_cancelled") if code == 130
                       else t("cli.model.unchanged"))
            self.redraw_session(message)
            return
        try:
            data = config.load(self.config_file)
        except ValueError as e:
            print(t("cli.config.error", error=e))
            return
        name, item = config.profile(data)
        if not item:
            print(t("cli.model.profile_missing"))
            return
        self.model = item["model"]
        self.thinking = item.get("thinking")
        self.num_ctx = item.get("num_ctx")
        self.provider = self._provider_from_profile(item)
        self._log("meta", event="model", profile=name, model=self.model,
                  thinking=self.thinking, num_ctx=self.num_ctx)
        context = (t("cli.model.context_suffix", context=_short_context(self.num_ctx))
                   if self.num_ctx else "")
        effort = (t("cli.model.effort_suffix", effort=self.thinking)
                  if self.thinking not in (None, False) else t("cli.model.no_reasoning"))
        self.redraw_session(
            t("cli.model.summary", name=name, context=context, effort=effort))

    def _permission_mode_label(self):
        return (t("cli.mode.saved_only") if self.permission_mode == "authorized_only"
                else t("cli.mode.safe_auto"))

    def _engine_label(self):
        if self.provider.get("provider") == "ollama":
            version = self.ensure_ollama(warn=False)
            return f"Ollama {version}" if version else t("cli.engine.unavailable")
        return self.provider.get("provider_name") or t("cli.engine.openai_compatible")

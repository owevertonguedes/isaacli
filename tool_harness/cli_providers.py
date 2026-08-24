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

    def release_kaggle_session(self, keeping=None):
        """End the Kaggle kernel this window opened, when it is no longer the one
        being talked to.

        A kernel spends quota by wall clock until something deletes it, and the
        quota is 30 hours a week. Until this existed the only end was leaving the
        program: switching to a local model kept the kernel billing for a model
        nothing was going to ask again. `keeping` is the profile being switched
        to, so re-selecting the kernel already held does not end it.

        The decision of whether it really stops is stop_profile_session's, not
        this one's: another window may be talking to the same kernel, and the
        last one out is what ends it.
        """
        held = getattr(self, "kaggle_profile", None)
        if not held or held == keeping:
            return
        import cli_kaggle

        cli_kaggle.stop_profile_session(held, self.config_file)
        self.kaggle_profile = None

    def apply_profile(self, item, name=None):
        """Everything a chosen profile decides about the next request.

        The five move together or not at all. They were assigned one by one in
        four places (`/model`, `/model <name>`, `/setup`, `/kaggle`), which is
        four chances to forget one: a profile that carries a temperature and a
        path that does not read it is a setting the user chose and the program
        silently ignores.

        Absent is a decision here, not a missing value. `None` means the
        profile did not choose, and the agent's own default holds; that is why
        this reads every field rather than updating only the ones present.

        Releasing whatever was serving the old model belongs here for the same
        reason: picking another model says the old one is done, and the four
        callers are four chances to forget it. Left running, llama-server or
        Ollama stayed resident with a model nothing would ask again, holding the
        card while the next engine tried to load onto it, and a Kaggle kernel
        went on spending quota by wall clock until the program was closed.

        `name` is the profile being applied, and it is what keeps the Kaggle
        release from ending the kernel it was just handed.
        """
        provider = self._provider_from_profile(item)
        # Re-picking what is already loaded releases nothing: stopping a server
        # only to start it again is a minute of loading for no change. The
        # release runs before the fields move, because the registration is keyed
        # on the provider being replaced.
        if item.get("model") != getattr(self, "model", None) or provider != getattr(
                self, "provider", None):
            self.release_local_server()
        self.release_kaggle_session(keeping=name)
        self.model = item["model"]
        self.thinking = item.get("thinking")
        self.num_ctx = item.get("num_ctx")
        self.temperature = item.get("temperature")
        self.provider = provider

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

        code = setup_ollama.run_model_selector(
            config_file=self.config_file, release_fn=self.release_local_server)
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
        self.apply_profile(item, name=name)
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

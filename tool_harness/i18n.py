"""Carregamento pequeno e padrao de catalogos de traducao JSON."""
import json
from pathlib import Path


LOCALES_DIR = Path(__file__).resolve().parent / "locales"
DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = {
    "pt-BR": "Português (Brasil)",
    "en": "English",
}


class Translator:
    def __init__(self, language=DEFAULT_LANGUAGE):
        self.language = language if language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
        self._fallback = self._load("en")
        self._messages = self._load(self.language)

    def _load(self, language):
        caminho = LOCALES_DIR / f"{language}.json"
        try:
            dado = json.loads(caminho.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"invalid translation catalog {caminho}: {e}") from e
        if not isinstance(dado, dict):
            raise RuntimeError(f"invalid translation catalog {caminho}: expected object")
        return dado

    def t(self, key, **values):
        template = self._messages.get(key, self._fallback.get(key, key))
        return template.format(**values)

"""Process-wide translator for cli.py and its collaborators.

One translator per process: the CLI is a single session, so threading a
Translator through every helper would be ceremony without a second reader.
"""
from i18n import Translator

_TRANSLATOR = Translator()


def set_language(code):
    global _TRANSLATOR
    _TRANSLATOR = Translator(code)
    return _TRANSLATOR


def t(key, **values):
    return _TRANSLATOR.t(key, **values)

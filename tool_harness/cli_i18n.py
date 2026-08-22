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


def translator():
    """The one translator of this process.

    Collaborators that hand a translator to a helper must ask for it here.
    Building `Translator()` on the spot silently yields English, which is how
    the quantization screen came out in English inside a Portuguese session.
    """
    return _TRANSLATOR


def t(key, **values):
    return _TRANSLATOR.t(key, **values)

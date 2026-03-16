"""Scope plugin registration for AI Transition Mixer."""

try:
    from scope.core.plugins.interface import hookimpl
except ImportError:
    def hookimpl(func):
        return func

from .pipeline import AiTransitionMixerPreprocessor


@hookimpl
def register_preprocessors():
    return [AiTransitionMixerPreprocessor]

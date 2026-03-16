"""Scope plugin registration for AI Transition Mixer."""

try:
    from scope.core.plugins import hookimpl
except ImportError:
    try:
        from scope.core.plugins.interface import hookimpl
    except ImportError:
        def hookimpl(func):
            return func

from .pipeline import AiTransitionMixerPreprocessor


@hookimpl
def register_pipelines(register):
    """Register the AI Transition Mixer preprocessor with Scope."""
    register(AiTransitionMixerPreprocessor)

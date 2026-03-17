"""Scope plugin registration for AI Transition Mixer."""

try:
    from scope.core.plugins import hookimpl
except ImportError:
    try:
        from scope.core.plugins.interface import hookimpl
    except ImportError:
        def hookimpl(func):
            return func

from .pipeline import AiTransitionMixerPipeline


@hookimpl
def register_pipelines(register):
    """Register the AI Transition Mixer graph node with Scope."""
    register(AiTransitionMixerPipeline)

"""Offline, deterministic replay metrics for agent_v1 artifacts."""

__all__ = ["benchmark_artifacts", "load_artifact_metrics"]


def benchmark_artifacts(*args, **kwargs):
    from .replay import benchmark_artifacts as implementation

    return implementation(*args, **kwargs)


def load_artifact_metrics(*args, **kwargs):
    from .replay import load_artifact_metrics as implementation

    return implementation(*args, **kwargs)

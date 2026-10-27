"""Numeric policy deployment is dependency-free; training needs the sim extra."""
from .artifact import load_policy, save_artifact, read_artifact, LearnedPolicy


def train(**kwargs):
    from .search import train as implementation
    return implementation(**kwargs)


def evaluate(parameters_or_artifact, **kwargs):
    from .rollout import evaluate as implementation
    return implementation(parameters_or_artifact, **kwargs)


__all__ = ['train','evaluate','load_policy','save_artifact','read_artifact','LearnedPolicy']

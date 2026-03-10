"""Walker generation pipeline — config-driven transition clip generation."""

from walker_generation.config import GenerationConfig, LongSequenceConfig, load_config


def run_generation(config: GenerationConfig) -> None:
    from walker_generation.generate import run_generation as _run

    _run(config)


__all__ = ["GenerationConfig", "LongSequenceConfig", "load_config", "run_generation"]

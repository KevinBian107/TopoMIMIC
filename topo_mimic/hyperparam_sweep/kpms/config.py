"""KPMS-specific hyperparameter configuration."""

from dataclasses import dataclass
from typing import Any


@dataclass
class KPMSHyperparams:
    kappa: float = 1e5
    latent_dim: int = 10
    num_states: int = 20
    ar_iters: int = 50
    full_iters: int = 300
    model_type: str = "arhmm"


def hyperparams_to_dict(hp: KPMSHyperparams) -> dict[str, Any]:
    return {
        "kappa": hp.kappa,
        "latent_dim": hp.latent_dim,
        "num_states": hp.num_states,
        "ar_iters": hp.ar_iters,
        "full_iters": hp.full_iters,
        "model_type": hp.model_type,
    }

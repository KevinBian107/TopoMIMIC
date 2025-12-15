"""
KPMS-based state space model clustering methods.
AR-HMM and SLDS for temporal segmentation and syllable assignment.
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp
import keypoint_moseq as kpms
from jax_moseq.models.keypoint_slds import estimate_coordinates
from jax_moseq.utils import unbatch


@dataclass
class KPMSConfig:
    latent_dim: int = 10
    kappa: float = 1e5
    ar_iters: int = 50
    full_iters: int = 300
    seed: int = 0


@dataclass
class KPMSResult:
    labels: np.ndarray
    n_clusters: int
    method_name: str
    model: Any
    results: dict
    reconstruction: np.ndarray | None = None


class BaseKPMSMethod(ABC):
    """Base class for KPMS-based clustering methods."""

    def __init__(self, name: str, config: KPMSConfig = None):
        self.name = name
        self.config = config or KPMSConfig()
        self._model = None
        self._results = None
        self._project_dir = None

    @abstractmethod
    def fit(
        self,
        data: np.ndarray,
        n_clusters: int,
        project_dir: str,
    ) -> KPMSResult:
        pass

    def _prepare_coordinates(
        self,
        data: np.ndarray,
    ) -> tuple[dict, dict]:
        """
        Prepare data in KPMS format.

        Args:
            data: Input array of shape (n_clips, n_frames, n_dim)

        Returns:
            Tuple of (coordinates, confidences) dicts
        """
        n_clips, n_frames, n_dim = data.shape
        coordinates = {}
        confidences = {}

        for i in range(n_clips):
            coords = data[i][:, :, np.newaxis].astype(np.float64)
            coordinates[f"recording_{i:04d}"] = coords
            confidences[f"recording_{i:04d}"] = np.ones((n_frames, n_dim), dtype=np.float64)

        return coordinates, confidences

    def _setup_project(self, project_dir: str, n_dim: int, num_states: int) -> None:
        """Setup KPMS project directory and config."""
        import shutil

        project_path = Path(project_dir)
        if project_path.exists():
            shutil.rmtree(project_path)
        project_path.mkdir(parents=True, exist_ok=True)

        kpms.setup_project(
            project_dir,
            fix_heading=True,
            bodyparts=['placeholder'],
            skeleton=[],
            fps=30,
            overwrite=True,
        )

        kpms.update_config(
            project_dir,
            fix_heading=False,
            bodyparts=[f'dim_{i}' for i in range(n_dim)],
            use_bodyparts=[f'dim_{i}' for i in range(n_dim)],
            anterior_bodyparts=['dim_0'],
            posterior_bodyparts=[f'dim_{n_dim-1}'],
            verbose=False,
            latent_dim=self.config.latent_dim,
            num_states=num_states,
            kappa=self.config.kappa,
        )

    def _extract_syllables(self, results: dict) -> np.ndarray:
        """Extract syllable labels from results dict."""
        rec_names = sorted(results.keys())
        all_syllables = []

        for rn in rec_names:
            syll = results[rn]['syllable']
            all_syllables.append(syll)

        return all_syllables

    def _compute_reconstruction(
        self,
        model: dict,
        results: dict,
        rec_name: str,
    ) -> np.ndarray:
        """Compute reconstructed signal from model."""
        Cd = model['params']['Cd']
        rr = results[rec_name]
        x = jnp.array(rr['latent_state'])
        v = jnp.array(rr['centroid'])
        h = jnp.array(rr['heading'])

        Y_hat = estimate_coordinates(x, v, h, jnp.array(Cd))
        Y_hat = jnp.array(Y_hat)
        if Y_hat.ndim == 3 and Y_hat.shape[-1] == 1:
            Y_hat = Y_hat[..., 0]

        return np.asarray(Y_hat)


class ARHMMClustering(BaseKPMSMethod):
    """
    AR-HMM clustering using keypoint-moseq.
    Uses ar_only=True for autoregressive HMM without latent state dynamics.
    """

    def __init__(self, config: KPMSConfig = None):
        super().__init__("arhmm", config)

    def fit(
        self,
        data: np.ndarray,
        n_clusters: int,
        project_dir: str,
    ) -> KPMSResult:
        """
        Fit AR-HMM model to data.

        Args:
            data: Input array (n_clips, n_frames, n_dim)
            n_clusters: Number of syllables/states
            project_dir: Directory for KPMS project files

        Returns:
            KPMSResult with syllable labels and model
        """
        n_clips, n_frames, n_dim = data.shape
        self._project_dir = project_dir

        self._setup_project(project_dir, n_dim, n_clusters)
        config = kpms.load_config(project_dir)

        coordinates, confidences = self._prepare_coordinates(data)
        int_data, int_metadata = kpms.format_data(coordinates, confidences, **config)

        model = kpms.init_model(
            data=int_data,
            metadata=int_metadata,
            states=None,
            **config,
            seed=jax.random.PRNGKey(self.config.seed + n_clusters),
        )
        model = kpms.update_hypparams(model, kappa=self.config.kappa)

        model_name = f'arhmm_states{n_clusters}'

        model = kpms.fit_model(
            model=model,
            data=int_data,
            metadata=int_metadata,
            project_dir=project_dir,
            model_name=model_name,
            ar_only=True,
            num_iters=self.config.ar_iters,
            num_states=n_clusters,
            generate_progress_plots=False,
        )[0]

        kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
        model_ckpt, _, _, _ = kpms.load_checkpoint(project_dir, model_name)
        model_ckpt['project_dir'] = project_dir
        model_ckpt['model_name'] = model_name

        results = kpms.extract_results(model_ckpt, int_metadata, project_dir, model_name)

        self._model = model_ckpt
        self._results = results

        all_syllables = self._extract_syllables(results)

        return KPMSResult(
            labels=all_syllables,
            n_clusters=n_clusters,
            method_name=self.name,
            model=model_ckpt,
            results=results,
        )


class SLDSClustering(BaseKPMSMethod):
    """
    SLDS clustering using keypoint-moseq.
    Uses ar_only=False for full switching linear dynamical system.
    """

    def __init__(self, config: KPMSConfig = None):
        super().__init__("slds", config)

    def fit(
        self,
        data: np.ndarray,
        n_clusters: int,
        project_dir: str,
    ) -> KPMSResult:
        """
        Fit SLDS model to data.

        Args:
            data: Input array (n_clips, n_frames, n_dim)
            n_clusters: Number of syllables/states
            project_dir: Directory for KPMS project files

        Returns:
            KPMSResult with syllable labels and model
        """
        n_clips, n_frames, n_dim = data.shape
        self._project_dir = project_dir

        self._setup_project(project_dir, n_dim, n_clusters)
        config = kpms.load_config(project_dir)

        coordinates, confidences = self._prepare_coordinates(data)
        int_data, int_metadata = kpms.format_data(coordinates, confidences, **config)

        model = kpms.init_model(
            data=int_data,
            metadata=int_metadata,
            states=None,
            **config,
            seed=jax.random.PRNGKey(self.config.seed + n_clusters),
        )
        model = kpms.update_hypparams(model, kappa=self.config.kappa)

        model_name = f'slds_states{n_clusters}'

        model = kpms.fit_model(
            model=model,
            data=int_data,
            metadata=int_metadata,
            project_dir=project_dir,
            model_name=model_name,
            ar_only=True,
            num_iters=self.config.ar_iters,
            num_states=n_clusters,
            generate_progress_plots=False,
        )[0]

        model = kpms.update_hypparams(model, kappa=self.config.kappa)

        kpms.fit_model(
            model=model,
            data=int_data,
            metadata=int_metadata,
            project_dir=project_dir,
            model_name=model_name,
            ar_only=False,
            start_iter=self.config.ar_iters,
            num_iters=self.config.full_iters,
            generate_progress_plots=False,
        )

        kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
        model_ckpt, _, _, _ = kpms.load_checkpoint(project_dir, model_name)
        model_ckpt['project_dir'] = project_dir
        model_ckpt['model_name'] = model_name

        results = kpms.extract_results(model_ckpt, int_metadata, project_dir, model_name)

        self._model = model_ckpt
        self._results = results

        all_syllables = self._extract_syllables(results)

        return KPMSResult(
            labels=all_syllables,
            n_clusters=n_clusters,
            method_name=self.name,
            model=model_ckpt,
            results=results,
        )


def plot_reconstruction_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    syllables: np.ndarray,
    title: str,
    save_path: str,
    n_dims: int = 3,
) -> None:
    """
    Plot original vs reconstructed signal for the first n_dims dimensions.

    Args:
        original: Original signal (n_frames, n_dim)
        reconstructed: Reconstructed signal (n_frames, n_dim)
        syllables: Syllable assignments (n_frames,)
        title: Plot title
        save_path: Path to save the figure
        n_dims: Number of dimensions to plot
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)

    n_dims = min(n_dims, original.shape[1], reconstructed.shape[1])
    timesteps = np.arange(len(original))

    unique_sylls = np.unique(syllables)
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_sylls))))
    syll_to_color = {s: colors[i] for i, s in enumerate(unique_sylls)}

    fig, axes = plt.subplots(n_dims + 1, 1, figsize=(14, 2 + n_dims * 2), sharex=True)

    for dim in range(n_dims):
        ax = axes[dim]
        ax.plot(timesteps, original[:, dim], linewidth=1.5, color='#2E86AB',
                alpha=0.85, label='Original')
        ax.plot(timesteps, reconstructed[:, dim], linewidth=1.5, linestyle='--',
                color='#A23B72', alpha=0.85, label='Reconstructed')

        changes = np.where(np.diff(syllables) != 0)[0] + 1
        changes = np.concatenate([[0], changes, [len(syllables)]])
        for i in range(len(changes) - 1):
            start = changes[i]
            end = min(changes[i + 1], len(timesteps))
            sid = syllables[start]
            ax.axvspan(start, end, alpha=0.15, color=syll_to_color[sid])

        if dim == 0:
            ax.legend(loc='upper right', fontsize=9)
        ax.set_ylabel(f'Dim {dim}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    ax_syll = axes[-1]
    changes = np.where(np.diff(syllables) != 0)[0] + 1
    changes = np.concatenate([[0], changes, [len(syllables)]])
    for i in range(len(changes) - 1):
        start = changes[i]
        end = min(changes[i + 1], len(timesteps))
        sid = syllables[start]
        ax_syll.barh(0, end - start, left=start, height=0.8,
                     color=syll_to_color[sid], edgecolor='black', linewidth=0.5)

    ax_syll.set_xlim(0, len(timesteps))
    ax_syll.set_ylim(-0.5, 0.5)
    ax_syll.set_yticks([])
    ax_syll.set_ylabel('Syllables', fontsize=10, fontweight='bold')
    ax_syll.set_xlabel('Timestep', fontsize=11, fontweight='bold')

    mse = np.mean((original[:, :n_dims] - reconstructed[:, :n_dims]) ** 2)
    fig.suptitle(f'{title}  |  MSE={mse:.6f}', fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def get_kpms_method(name: str, config: KPMSConfig = None) -> BaseKPMSMethod:
    """Factory function to get KPMS method by name."""
    methods = {
        "arhmm": ARHMMClustering,
        "slds": SLDSClustering,
    }

    if name not in methods:
        raise ValueError(f"Unknown KPMS method: {name}. Available: {list(methods.keys())}")

    return methods[name](config)

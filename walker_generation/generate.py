"""Core generation pipeline: schedule -> rollout -> crop -> extract -> save.

Wraps walker/scripts/generate_topo_dataset.py (Step 1) and
walker/scripts/extract_imitation_activations.py (Step 2) into a single
config-driven pipeline that outputs directly to TopoVNL format.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
import time
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from walker_generation.config import GenerationConfig, TransitionSpec
from walker_generation.output import write_labels_csv, write_unified_h5

# Imports from walker/scripts/ (must be on sys.path or installed)
from walker.scripts.generate_topo_dataset import (
    ClipMeta,
    TransitionMeta,
    build_inference_fn,
    crop_clips_from_rollout,
    load_step1_checkpoint,
    make_transition_schedule,
)
from walker.scripts.extract_imitation_activations import (
    SingleClipGenerator,
    build_imitation_env,
    make_batched_rollout_fn,
)

from vnl_playground.tasks.walker import consts
from vnl_playground.tasks.walker.multi_behavior import (
    MultiBehaviorWalker,
    default_config as multi_behavior_config,
)
from vnl_playground.tasks.walker.online_reference import (
    OnlineReferenceGenerator,
    WalkerTrajectory,
)
from track_mjx.agent.checkpointing import load_checkpoint_for_eval, load_inference_fn


def _resolve_mode_index(name: str) -> int:
    """Map a behavior mode name to its integer index."""
    if name not in consts.BEHAVIOR_MODES:
        raise ValueError(
            f"Unknown behavior mode {name!r}. "
            f"Available: {list(consts.BEHAVIOR_MODES.keys())}"
        )
    return consts.BEHAVIOR_MODES[name]


def _generate_clips_for_pair(
    spec: TransitionSpec,
    batched_generate,
    np_rng: np.random.RandomState,
    jax_rng,
    config: GenerationConfig,
) -> tuple[list[dict], list[ClipMeta], int]:
    """Generate clips for a single transition pair via targeted 2-mode scheduling.

    Restricts allowed_modes to [src, dst] so every rollout MUST alternate
    between those two modes. Filters cropped clips to keep only the
    requested src->dst direction.

    Args:
        spec: TransitionSpec with src, dst, n_clips.
        batched_generate: JIT+vmap'd Step 1 generation function.
        np_rng: NumPy random state for schedule generation.
        jax_rng: JAX PRNGKey for rollout RNG splitting.
        config: Full generation config.

    Returns:
        Tuple of (clips, clip_metas, jax_rng) where clips/metas are the
        filtered src->dst clips and jax_rng is the updated key.
    """
    src_idx = _resolve_mode_index(spec.src)
    dst_idx = _resolve_mode_index(spec.dst)
    allowed_modes = [src_idx, dst_idx]

    sched = config.schedule
    batch_n = config.rollout.batch_size

    collected_clips: list[dict] = []
    collected_metas: list[ClipMeta] = []
    total_rollouts = 0

    print(f"\n  Pair {spec.src} -> {spec.dst}: target {spec.n_clips} clips")

    while len(collected_clips) < spec.n_clips:
        batch_schedules = []
        batch_transitions = []
        for _ in range(batch_n):
            schedule, transitions = make_transition_schedule(
                np_rng,
                sched.rollout_frames,
                allowed_modes,
                mode_duration_min=sched.mode_duration_min,
                mode_duration_extra_max=sched.mode_duration_extra_max,
                transition_mean=sched.transition_mean,
                transition_std=sched.transition_std,
            )
            batch_schedules.append(schedule)
            batch_transitions.append(transitions)

        stacked_schedules = jp.stack([jp.array(s) for s in batch_schedules])

        batch_rngs = jax.random.split(jax_rng, batch_n + 1)
        jax_rng = batch_rngs[0]
        gen_rngs = batch_rngs[1:]

        batch_trajs = batched_generate(gen_rngs, stacked_schedules)

        for i in range(batch_n):
            np_traj = WalkerTrajectory(
                qpos=np.asarray(batch_trajs.qpos[i]),
                qvel=np.asarray(batch_trajs.qvel[i]),
                xpos=np.asarray(batch_trajs.xpos[i]),
                xquat=np.asarray(batch_trajs.xquat[i]),
                behavior_labels=np.asarray(batch_trajs.behavior_labels[i]),
            )
            clips, clip_metas = crop_clips_from_rollout(
                np_traj, batch_transitions[i], sched.clip_length
            )
            for clip, meta in zip(clips, clip_metas):
                if meta.src_mode == src_idx and meta.dst_mode == dst_idx:
                    collected_clips.append(clip)
                    collected_metas.append(meta)

        total_rollouts += batch_n
        print(
            f"    {len(collected_clips)}/{spec.n_clips} clips "
            f"({total_rollouts} rollouts)"
        )

    collected_clips = collected_clips[: spec.n_clips]
    collected_metas = collected_metas[: spec.n_clips]

    return collected_clips, collected_metas, jax_rng


def _extract_activations(
    clips: list[dict],
    config: GenerationConfig,
) -> dict[str, np.ndarray]:
    """Run all clips through Step 2 policy and extract activations.

    Args:
        clips: List of clip dicts (qpos, qvel, xpos, xquat, behavior_labels).
        config: Full generation config.

    Returns:
        Dict with "latent_mean" (N, T, 16) and "agent_qpos" (N, T, 9).
    """
    n_clips = len(clips)
    clip_length = config.schedule.clip_length
    batch_size = config.rollout.extraction_batch_size

    print(f"\nLoading Step 2 checkpoint from {config.checkpoints.step2_path}")
    ckpt = load_checkpoint_for_eval(config.checkpoints.step2_path)
    step2_cfg = ckpt["cfg"]
    step2_policy_params = ckpt["policy"]

    reference_length = step2_cfg.env_config.reference_length
    episode_length = clip_length - reference_length
    print(f"  reference_length: {reference_length}")
    print(f"  episode_length:   {episode_length}")

    policy_fn = load_inference_fn(
        step2_cfg,
        policy_params=step2_policy_params,
        deterministic=True,
        get_activation=True,
    )

    print(f"Building imitation environment (Step 1: {config.checkpoints.step1_path})")
    env = build_imitation_env(
        step1_checkpoint=config.checkpoints.step1_path,
        clip_length=clip_length,
        reference_length=reference_length,
    )

    print(f"Building batched rollout function (batch_size={batch_size})...")
    batched_rollout = make_batched_rollout_fn(env, policy_fn, episode_length)

    # Convert clips to JAX arrays
    jax_clips = []
    for c in clips:
        jax_clips.append(WalkerTrajectory(
            qpos=jp.array(c["qpos"]),
            qvel=jp.array(c["qvel"]),
            xpos=jp.array(c["xpos"]),
            xquat=jp.array(c["xquat"]),
            behavior_labels=jp.array(c["behavior_labels"]),
        ))

    # JIT warmup
    print("JIT compiling batched rollout...")
    rng = jax.random.PRNGKey(config.rollout.seed + 1000)
    warmup_qpos = jp.stack([jax_clips[0].qpos] * batch_size)
    warmup_qvel = jp.stack([jax_clips[0].qvel] * batch_size)
    warmup_xpos = jp.stack([jax_clips[0].xpos] * batch_size)
    warmup_xquat = jp.stack([jax_clips[0].xquat] * batch_size)
    warmup_labels = jp.stack([jax_clips[0].behavior_labels] * batch_size)
    warmup_rngs = jax.random.split(rng, batch_size + 1)
    rng = warmup_rngs[0]
    warmup_data = batched_rollout(
        warmup_qpos, warmup_qvel, warmup_xpos, warmup_xquat,
        warmup_labels, warmup_rngs[1:],
    )
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), warmup_data)
    print("JIT compilation done.")

    # Process all clips in batches
    print(f"Extracting activations for {n_clips} clips...")
    t_start = time.time()

    all_latent_mean = []
    all_agent_qpos = []

    for batch_start in range(0, n_clips, batch_size):
        batch_end = min(batch_start + batch_size, n_clips)
        batch_n = batch_end - batch_start

        batch_qpos = jp.stack([jax_clips[i].qpos for i in range(batch_start, batch_end)])
        batch_qvel = jp.stack([jax_clips[i].qvel for i in range(batch_start, batch_end)])
        batch_xpos = jp.stack([jax_clips[i].xpos for i in range(batch_start, batch_end)])
        batch_xquat = jp.stack([jax_clips[i].xquat for i in range(batch_start, batch_end)])
        batch_labels = jp.stack(
            [jax_clips[i].behavior_labels for i in range(batch_start, batch_end)]
        )

        if batch_n < batch_size:
            pad_n = batch_size - batch_n
            batch_qpos = jp.concatenate(
                [batch_qpos, jp.zeros((pad_n, *batch_qpos.shape[1:]))], axis=0
            )
            batch_qvel = jp.concatenate(
                [batch_qvel, jp.zeros((pad_n, *batch_qvel.shape[1:]))], axis=0
            )
            batch_xpos = jp.concatenate(
                [batch_xpos, jp.zeros((pad_n, *batch_xpos.shape[1:]))], axis=0
            )
            batch_xquat = jp.concatenate(
                [batch_xquat, jp.zeros((pad_n, *batch_xquat.shape[1:]))], axis=0
            )
            batch_labels = jp.concatenate(
                [batch_labels, jp.zeros((pad_n, *batch_labels.shape[1:]))], axis=0
            )

        batch_rngs = jax.random.split(rng, batch_size + 1)
        rng = batch_rngs[0]
        batch_data = batched_rollout(
            batch_qpos, batch_qvel, batch_xpos, batch_xquat,
            batch_labels, batch_rngs[1:],
        )

        if batch_n < batch_size:
            batch_data = jax.tree_util.tree_map(lambda x: x[:batch_n], batch_data)

        all_latent_mean.append(np.asarray(batch_data["latent_mean"]))
        all_agent_qpos.append(np.asarray(batch_data["agent_qpos"]))

        elapsed = time.time() - t_start
        clips_done = batch_end
        print(
            f"  Batch {batch_start // batch_size + 1}: "
            f"clips {batch_start + 1}-{batch_end}/{n_clips} "
            f"({clips_done / elapsed:.1f} clips/s)"
        )

    elapsed = time.time() - t_start
    print(f"Extraction done: {n_clips} clips in {elapsed:.1f}s")

    return {
        "latent_mean": np.concatenate(all_latent_mean, axis=0),
        "agent_qpos": np.concatenate(all_agent_qpos, axis=0),
    }


def run_generation(config: GenerationConfig) -> None:
    """Run the full generation pipeline.

    1. Load Step 1 checkpoint and build batched generator
    2. For each TransitionSpec, generate targeted clips
    3. Extract activations via Step 2 policy
    4. Write unified output (H5 + CSV)

    Args:
        config: Full generation configuration.
    """
    total_clips = sum(t.n_clips for t in config.transitions)
    print(f"Walker generation pipeline")
    print(f"  {len(config.transitions)} transition pairs, {total_clips} total clips")

    # ── Step 1: Load checkpoint and build generator ─────────────────────
    print(f"\nLoading Step 1 checkpoint from {config.checkpoints.step1_path}")
    step1_params = load_step1_checkpoint(
        config.checkpoints.step1_path,
        config.checkpoints.step1_step,
    )

    cfg = multi_behavior_config()
    cfg.mujoco_impl = "warp"
    gen_env = MultiBehaviorWalker(config=cfg)

    inference_fn = build_inference_fn(
        step1_params,
        obs_size=gen_env.observation_size,
        action_size=gen_env.action_size,
    )

    sched = config.schedule
    generator = OnlineReferenceGenerator(
        policy_fn=inference_fn,
        walker_env=gen_env,
        n_frames=sched.rollout_frames,
        deterministic=True,
        warmup_frames=sched.warmup_frames,
        warmup_transition_frames=sched.warmup_transition_frames,
    )

    batch_n = config.rollout.batch_size
    batched_generate = jax.jit(jax.vmap(generator.generate, in_axes=(0, 0)))

    # JIT warmup
    print(f"JIT compiling Step 1 batched generate (batch_size={batch_n})...")
    np_rng = np.random.RandomState(config.rollout.seed)
    warmup_schedule, _ = make_transition_schedule(
        np_rng,
        sched.rollout_frames,
        list(range(consts.N_BEHAVIOR_MODES)),
        mode_duration_min=sched.mode_duration_min,
        mode_duration_extra_max=sched.mode_duration_extra_max,
        transition_mean=sched.transition_mean,
        transition_std=sched.transition_std,
    )
    warmup_schedules = jp.stack([jp.array(warmup_schedule)] * batch_n)
    warmup_rngs = jax.random.split(
        jax.random.PRNGKey(config.rollout.seed + 999), batch_n
    )
    _ = batched_generate(warmup_rngs, warmup_schedules)
    print("JIT compilation done.")

    # ── Step 2: Generate clips per transition pair ──────────────────────
    np_rng = np.random.RandomState(config.rollout.seed)
    jax_rng = jax.random.PRNGKey(config.rollout.seed)

    all_clips: list[dict] = []
    all_clip_metas: list[ClipMeta] = []

    t_start = time.time()
    for spec in config.transitions:
        clips, metas, jax_rng = _generate_clips_for_pair(
            spec, batched_generate, np_rng, jax_rng, config
        )
        all_clips.extend(clips)
        all_clip_metas.extend(metas)

    elapsed = time.time() - t_start
    print(f"\nStep 1 done: {len(all_clips)} clips in {elapsed:.1f}s")

    _print_pair_summary(all_clip_metas)

    # ── Step 3: Extract activations ─────────────────────────────────────
    activations = _extract_activations(all_clips, config)

    # ── Step 4: Write output ────────────────────────────────────────────
    print(f"\nWriting output...")
    write_unified_h5(
        config.output.h5_path,
        latent_mean=activations["latent_mean"],
        agent_qpos=activations["agent_qpos"],
    )

    src_modes = np.array([m.src_mode for m in all_clip_metas], dtype=np.int32)
    dst_modes = np.array([m.dst_mode for m in all_clip_metas], dtype=np.int32)
    t_starts = np.array(
        [m.transition_start_in_clip for m in all_clip_metas], dtype=np.int32
    )
    t_ends = np.array(
        [m.transition_end_in_clip for m in all_clip_metas], dtype=np.int32
    )
    write_labels_csv(config.output.csv_path, src_modes, dst_modes, t_starts, t_ends)

    print(f"\nPipeline complete.")
    print(f"  H5:  {config.output.h5_path}")
    print(f"  CSV: {config.output.csv_path}")


def _print_pair_summary(clip_metas: list[ClipMeta]) -> None:
    """Print a summary of collected transition pairs."""
    from walker_generation.output import WALKER_MODES

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for m in clip_metas:
        pair_counts[(m.src_mode, m.dst_mode)] += 1

    print("\nCollected transition pairs:")
    for (src, dst), count in sorted(pair_counts.items()):
        src_name = WALKER_MODES[src]
        dst_name = WALKER_MODES[dst]
        print(f"  {src_name:>15s} -> {dst_name:<15s}: {count}")
    print(f"  Total: {len(clip_metas)} clips")

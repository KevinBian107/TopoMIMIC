"""Core generation pipeline: schedule -> rollout -> crop -> extract -> save.

Generates targeted walker behavior transition clips and extracts Step 2
policy activations, outputting directly to TopoVNL format (H5 + CSV).

Requires the track-mjx environment (vnl_playground, track_mjx, brax, jax).
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import jax
import jax.numpy as jp
import numpy as np
import orbax.checkpoint as ocp
from brax.training import distribution
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as brax_ppo_networks

from vnl_playground.tasks.walker import consts
from vnl_playground.tasks.walker.multi_behavior import (
    MultiBehaviorWalker,
    default_config as multi_behavior_config,
)
from vnl_playground.tasks.walker.imitation import (
    WalkerImitation,
    default_config as imitation_default_config,
)
from vnl_playground.tasks.walker.online_reference import (
    OnlineReferenceGenerator,
    WalkerTrajectory,
)
from track_mjx.agent.checkpointing import load_checkpoint_for_eval, load_inference_fn

from walker_generation.config import GenerationConfig, LongSequenceConfig, TransitionSpec
from walker_generation.output import (
    WALKER_MODES,
    write_labels_csv,
    write_multi_segment_labels_csv,
    write_unified_h5,
)

N_MODES = consts.N_BEHAVIOR_MODES


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class TransitionMeta:
    src_mode: int
    dst_mode: int
    transition_start: int
    transition_end: int


@dataclass
class ClipMeta:
    src_mode: int
    dst_mode: int
    transition_start_in_clip: int
    transition_end_in_clip: int


class SingleClipGenerator:
    """Drop-in replacement for OnlineReferenceGenerator that returns a fixed clip."""

    def __init__(self, trajectory: WalkerTrajectory):
        self.trajectory = trajectory
        self.n_frames = trajectory.qpos.shape[0]

    def generate(self, rng, behavior_schedule):
        return self.trajectory

    def sample_behavior_schedule(self, rng, n_frames):
        return self.trajectory.behavior_labels


# ── Checkpoint loading ───────────────────────────────────────────────────────


def _load_step1_checkpoint(checkpoint_path: str, step: int | None = None):
    ckpt_dir = Path(checkpoint_path).resolve()
    if step is None:
        steps = sorted(
            int(d.name) for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()
        )
        if not steps:
            raise FileNotFoundError(f"No checkpoint steps found in {ckpt_dir}")
        step = steps[-1]

    step_path = str(ckpt_dir / str(step))
    print(f"Loading Step 1 checkpoint from {step_path}")
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(step_path)


def _build_step1_inference_fn(
    step1_params,
    obs_size: int,
    action_size: int,
    hidden_sizes: tuple[int, ...] = (256, 256, 256),
):
    ppo_networks = brax_ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=hidden_sizes,
        value_hidden_layer_sizes=hidden_sizes,
    )
    parametric_action_dist = distribution.NormalTanhDistribution(event_size=action_size)

    raw_normalizer = step1_params[0]
    policy_params = step1_params[1]

    if isinstance(raw_normalizer, dict):
        normalizer_params = running_statistics.RunningStatisticsState(
            count=raw_normalizer["count"],
            mean=raw_normalizer["mean"],
            summed_variance=raw_normalizer["summed_variance"],
            std=raw_normalizer["std"],
        )
    else:
        normalizer_params = raw_normalizer

    def inference_fn(obs, rng):
        logits = ppo_networks.policy_network.apply(
            normalizer_params, policy_params, obs
        )
        return parametric_action_dist.sample(logits, rng)

    return inference_fn


# ── Schedule generation ──────────────────────────────────────────────────────


def _make_transition_schedule(
    rng: np.random.RandomState,
    n_frames: int,
    allowed_modes: list[int],
    mode_duration_min: int = 250,
    mode_duration_extra_max: int = 100,
    transition_mean: int = 100,
    transition_std: int = 20,
) -> tuple[np.ndarray, list[TransitionMeta]]:
    schedule = np.zeros((n_frames, N_MODES), dtype=np.float32)
    transitions = []

    current_mode = rng.choice(allowed_modes)
    frame = 0

    while frame < n_frames:
        duration = mode_duration_min + rng.randint(0, mode_duration_extra_max + 1)
        seg_end = min(frame + duration, n_frames)
        schedule[frame:seg_end, current_mode] = 1.0
        frame = seg_end

        if frame >= n_frames:
            break

        next_mode = rng.choice(allowed_modes)
        while next_mode == current_mode:
            next_mode = rng.choice(allowed_modes)

        trans_dur_raw = rng.normal(transition_mean, transition_std)
        trans_dur_lo = transition_mean - 2 * transition_std
        trans_dur_hi = transition_mean + 2 * transition_std
        trans_dur = int(np.clip(trans_dur_raw, trans_dur_lo, trans_dur_hi))
        trans_dur = max(trans_dur, 1)

        trans_start = frame
        trans_end = min(frame + trans_dur, n_frames)
        actual_dur = trans_end - trans_start

        alphas = np.linspace(0.0, 1.0, actual_dur)
        for i, alpha in enumerate(alphas):
            schedule[trans_start + i, current_mode] = 1.0 - alpha
            schedule[trans_start + i, next_mode] = alpha

        transitions.append(
            TransitionMeta(
                src_mode=current_mode,
                dst_mode=next_mode,
                transition_start=trans_start,
                transition_end=trans_end,
            )
        )

        frame = trans_end
        current_mode = next_mode

    return schedule, transitions


# ── Long-sequence schedule ───────────────────────────────────────────────────


@dataclass
class LongSequenceClipMeta:
    segments: list[dict]
    transitions: list[TransitionMeta]


def _make_long_sequence_schedule(
    rng: np.random.RandomState,
    clip_length: int,
    n_transitions: int,
    source_pool_idx: list[int],
    target_pool_idx: list[int],
    transition_length: int = 100,
    variable: bool = False,
    trans_min: int = 50,
    trans_max: int = 150,
) -> tuple[np.ndarray, LongSequenceClipMeta]:
    all_modes = list(set(source_pool_idx + target_pool_idx))
    schedule = np.zeros((clip_length, N_MODES), dtype=np.float32)
    transitions = []
    segments = []

    total_trans_budget = 0
    trans_lengths = []
    for _ in range(n_transitions):
        if variable:
            tl = rng.randint(trans_min, trans_max + 1)
        else:
            tl = transition_length
        trans_lengths.append(tl)
        total_trans_budget += tl

    remaining = clip_length - total_trans_budget
    n_segments = n_transitions + 1
    if remaining < n_segments * 10:
        seg_dur_base = 10
    else:
        seg_dur_base = remaining // n_segments
    seg_durations = [seg_dur_base] * n_segments
    leftover = remaining - seg_dur_base * n_segments
    for i in range(max(leftover, 0)):
        seg_durations[i % n_segments] += 1

    mode_sequence = []
    current_mode = rng.choice(all_modes)
    mode_sequence.append(current_mode)
    for _ in range(n_transitions):
        next_mode = rng.choice(all_modes)
        while next_mode == current_mode and len(all_modes) > 1:
            next_mode = rng.choice(all_modes)
        mode_sequence.append(next_mode)
        current_mode = next_mode

    frame = 0
    for i in range(n_segments):
        seg_start = frame
        seg_end = min(frame + seg_durations[i], clip_length)
        mode = mode_sequence[i]
        schedule[seg_start:seg_end, mode] = 1.0
        segments.append({
            "behavior": WALKER_MODES[mode],
            "start_frame": seg_start,
            "end_frame": seg_end,
        })
        frame = seg_end

        if i < n_transitions and frame < clip_length:
            trans_dur = min(trans_lengths[i], clip_length - frame)
            trans_start = frame
            trans_end = frame + trans_dur

            src_mode = mode_sequence[i]
            dst_mode = mode_sequence[i + 1]
            alphas = np.linspace(0.0, 1.0, trans_dur)
            for j, alpha in enumerate(alphas):
                schedule[trans_start + j, src_mode] = 1.0 - alpha
                schedule[trans_start + j, dst_mode] = alpha

            transitions.append(TransitionMeta(
                src_mode=src_mode,
                dst_mode=dst_mode,
                transition_start=trans_start,
                transition_end=trans_end,
            ))
            frame = trans_end

    if frame < clip_length:
        last_mode = mode_sequence[-1]
        schedule[frame:clip_length, last_mode] = 1.0
        if segments:
            segments[-1]["end_frame"] = clip_length

    meta = LongSequenceClipMeta(segments=segments, transitions=transitions)
    return schedule, meta


def _generate_long_sequence_clips(
    batched_generate,
    np_rng: np.random.RandomState,
    jax_rng,
    config: GenerationConfig,
) -> tuple[list[dict], list[LongSequenceClipMeta], int]:
    ls = config.long_sequence
    sched = config.schedule
    batch_n = config.rollout.batch_size

    source_pool_idx = [_resolve_mode_index(m) for m in ls.source_pool]
    target_pool_idx = [_resolve_mode_index(m) for m in ls.target_pool]

    collected_clips: list[dict] = []
    collected_metas: list[LongSequenceClipMeta] = []

    print(f"\nGenerating {ls.n_clips} long-sequence clips "
          f"({ls.clip_length} frames, {ls.n_transitions} transitions each)")

    while len(collected_clips) < ls.n_clips:
        current_batch = min(batch_n, ls.n_clips - len(collected_clips))

        batch_schedules = []
        batch_metas = []
        for _ in range(current_batch):
            schedule, meta = _make_long_sequence_schedule(
                np_rng,
                clip_length=ls.clip_length,
                n_transitions=ls.n_transitions,
                source_pool_idx=source_pool_idx,
                target_pool_idx=target_pool_idx,
                transition_length=ls.transition_length,
                variable=ls.variable_transition_length,
                trans_min=ls.transition_length_min,
                trans_max=ls.transition_length_max,
            )
            if schedule.shape[0] < sched.rollout_frames:
                pad_len = sched.rollout_frames - schedule.shape[0]
                pad = np.tile(schedule[-1:], (pad_len, 1))
                schedule = np.concatenate([schedule, pad], axis=0)
            batch_schedules.append(schedule)
            batch_metas.append(meta)

        if current_batch < batch_n:
            for _ in range(batch_n - current_batch):
                batch_schedules.append(batch_schedules[0])

        stacked_schedules = jp.stack([jp.array(s) for s in batch_schedules])
        batch_rngs = jax.random.split(jax_rng, batch_n + 1)
        jax_rng = batch_rngs[0]

        batch_trajs = batched_generate(batch_rngs[1:], stacked_schedules)

        for i in range(current_batch):
            clip_end = ls.clip_length
            clip_data = {
                "qpos": np.asarray(batch_trajs.qpos[i])[:clip_end],
                "qvel": np.asarray(batch_trajs.qvel[i])[:clip_end],
                "xpos": np.asarray(batch_trajs.xpos[i])[:clip_end],
                "xquat": np.asarray(batch_trajs.xquat[i])[:clip_end],
                "behavior_labels": np.asarray(batch_trajs.behavior_labels[i])[:clip_end],
            }
            collected_clips.append(clip_data)
            collected_metas.append(batch_metas[i])

        print(f"  {len(collected_clips)}/{ls.n_clips} clips generated")

    return collected_clips[:ls.n_clips], collected_metas[:ls.n_clips], jax_rng


def _render_demo_clip(
    clip_data: dict,
    meta: LongSequenceClipMeta,
    video_path: str,
    xml_path: str = "topo_mimic/dmwk/xml/walker.xml",
    framerate: int = 30,
) -> None:
    import mujoco
    import imageio
    from PIL import Image, ImageDraw, ImageFont

    qpos = clip_data["qpos"]
    n_frames = qpos.shape[0]

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=240, width=320)

    try:
        font_label = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_trans = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font_label = ImageFont.load_default()
        font_trans = ImageFont.load_default()

    frames = []
    for fi in range(n_frames):
        mj_data.qpos[:] = qpos[fi]
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera="side")
        raw = renderer.render().copy()

        img = Image.fromarray(raw)
        draw = ImageDraw.Draw(img)

        behavior = None
        for seg in meta.segments:
            if seg["start_frame"] <= fi < seg["end_frame"]:
                behavior = seg["behavior"]
                break
        if behavior:
            draw.text((4, 4), behavior, font=font_label, fill=(255, 255, 255))

        for tr in meta.transitions:
            if tr.transition_start <= fi <= tr.transition_end:
                from walker_generation.output import WALKER_MODES
                src = WALKER_MODES[tr.src_mode]
                dst = WALKER_MODES[tr.dst_mode]
                label = f"{src} \u2192 {dst}"
                tw = draw.textbbox((0, 0), label, font=font_trans)[2]
                tx = max(0, (320 - tw) // 2)
                draw.rectangle(
                    (tx - 4, 220, tx + tw + 4, 238),
                    fill=(0, 0, 0, 160),
                )
                draw.text((tx, 221), label, font=font_trans, fill=(255, 100, 100))
                break

        frames.append(np.array(img))

    renderer.close()

    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(video_path, fps=framerate)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print(f"Saved demo clip video ({n_frames} frames): {video_path}")
    for seg in meta.segments:
        print(f"  {seg['behavior']:>15s}: frames {seg['start_frame']}-{seg['end_frame']}")


# ── Clip cropping ────────────────────────────────────────────────────────────


def _crop_clips_from_rollout(
    trajectory: WalkerTrajectory,
    transitions: list[TransitionMeta],
    clip_length: int,
) -> tuple[list[dict], list[ClipMeta]]:
    n_rollout_frames = trajectory.qpos.shape[0]
    clips = []
    clip_metas = []

    for trans in transitions:
        trans_dur = trans.transition_end - trans.transition_start
        pre_frames = (clip_length - trans_dur) // 2
        clip_start = trans.transition_start - pre_frames
        clip_end = clip_start + clip_length

        if clip_start < 0 or clip_end > n_rollout_frames:
            continue

        clip_data = {
            "qpos": trajectory.qpos[clip_start:clip_end],
            "qvel": trajectory.qvel[clip_start:clip_end],
            "xpos": trajectory.xpos[clip_start:clip_end],
            "xquat": trajectory.xquat[clip_start:clip_end],
            "behavior_labels": trajectory.behavior_labels[clip_start:clip_end],
        }
        clips.append(clip_data)

        clip_metas.append(
            ClipMeta(
                src_mode=trans.src_mode,
                dst_mode=trans.dst_mode,
                transition_start_in_clip=trans.transition_start - clip_start,
                transition_end_in_clip=trans.transition_end - clip_start,
            )
        )

    return clips, clip_metas


# ── Imitation environment ────────────────────────────────────────────────────


def _build_imitation_env(
    step1_checkpoint: str,
    clip_length: int,
    reference_length: int,
) -> WalkerImitation:
    mb_cfg = multi_behavior_config()
    mb_cfg.mujoco_impl = "warp"
    mb_env = MultiBehaviorWalker(config=mb_cfg)

    step1_params = _load_step1_checkpoint(step1_checkpoint)
    step1_fn = _build_step1_inference_fn(
        step1_params,
        obs_size=mb_env.observation_size,
        action_size=mb_env.action_size,
    )

    dummy_generator = OnlineReferenceGenerator(
        policy_fn=step1_fn,
        walker_env=mb_env,
        n_frames=clip_length,
        deterministic=True,
        warmup_frames=0,
        warmup_transition_frames=0,
    )

    imit_cfg = imitation_default_config()
    imit_cfg.trajectory_length = clip_length
    imit_cfg.reference_length = reference_length
    imit_cfg.warmup_frames = 0
    imit_cfg.warmup_transition_frames = 0
    imit_cfg.mujoco_impl = "warp"

    return WalkerImitation(config=imit_cfg, generator=dummy_generator)


def _flatten_nested_obs(nested):
    """Flatten a potentially nested observation to a single flat array.

    Handles both flat arrays and nested dicts/OrderedDicts of arrays.
    Preserves batch dimensions (leading dims shared across all leaves).
    """
    if isinstance(nested, jp.ndarray):
        return nested

    leaves = jax.tree_util.tree_leaves(nested)
    if not leaves:
        return jp.array([])

    ref = min(leaves, key=lambda x: x.ndim)
    n_batch = 0
    for i in range(ref.ndim - 1):
        if len({leaf.shape[i] for leaf in leaves}) == 1:
            n_batch += 1
        else:
            break

    if n_batch == 0:
        flat, _ = jax.flatten_util.ravel_pytree(nested)
        return flat

    batch_shape = ref.shape[:n_batch]
    return jp.concatenate(
        [leaf.reshape(*batch_shape, -1) for leaf in leaves], axis=-1
    )


def _flatten_obs(obs):
    """Flatten nested env observations to match the checkpoint normalizer structure.

    The WalkerImitation env produces deeply nested OrderedDicts:
        OrderedDict(state=OrderedDict(task_obs=OrderedDict(...), proprioception=OrderedDict(...)), ...)
    But the checkpoint normalizer expects flat arrays at each key:
        {state: {task_obs: flat_array, proprioception: flat_array}, ...}
    """
    result = {}
    for top_key in obs:
        inner = obs[top_key]
        result[top_key] = {
            "task_obs": _flatten_nested_obs(inner["task_obs"]),
            "proprioception": _flatten_nested_obs(inner["proprioception"]),
        }
    return result


def _make_batched_rollout_fn(env, policy_fn, episode_length):
    def rollout_one(clip_qpos, clip_qvel, clip_xpos, clip_xquat, clip_labels, rng):
        clip_traj = WalkerTrajectory(
            qpos=clip_qpos,
            qvel=clip_qvel,
            xpos=clip_xpos,
            xquat=clip_xquat,
            behavior_labels=clip_labels,
        )
        env.generator = SingleClipGenerator(clip_traj)

        rng, reset_rng = jax.random.split(rng)
        state = env.reset(reset_rng, behavior_schedule=clip_labels)

        def scan_step(carry, _):
            state, rng = carry
            rng, step_rng, policy_rng = jax.random.split(rng, 3)
            obs = _flatten_obs(state.obs)
            action, extras = policy_fn(obs, policy_rng)
            next_state = env.step(state, action)
            step_data = {
                "latent_mean": extras["latent_mean"],
                "latent_logvar": extras["latent_logvar"],
                "actions": action,
                "rewards": next_state.reward,
                "agent_qpos": next_state.data.qpos,
                "done": next_state.done,
            }
            return (next_state, rng), step_data

        _, traj_data = jax.lax.scan(
            scan_step, (state, rng), None, length=episode_length
        )
        return traj_data

    return jax.jit(jax.vmap(rollout_one, in_axes=(0, 0, 0, 0, 0, 0)))


# ── Per-pair clip generation ─────────────────────────────────────────────────


def _resolve_mode_index(name: str) -> int:
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
    """Generate clips for a single transition pair via targeted 2-mode scheduling."""
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
            schedule, transitions = _make_transition_schedule(
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
            clips, clip_metas = _crop_clips_from_rollout(
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


# ── Activation extraction ────────────────────────────────────────────────────


def _extract_activations(
    clips: list[dict],
    config: GenerationConfig,
) -> dict[str, np.ndarray]:
    """Run all clips through Step 2 policy and extract activations.

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
    env = _build_imitation_env(
        step1_checkpoint=config.checkpoints.step1_path,
        clip_length=clip_length,
        reference_length=reference_length,
    )

    print(f"Building batched rollout function (batch_size={batch_size})...")
    batched_rollout = _make_batched_rollout_fn(env, policy_fn, episode_length)

    jax_clips = [
        WalkerTrajectory(
            qpos=jp.array(c["qpos"]),
            qvel=jp.array(c["qvel"]),
            xpos=jp.array(c["xpos"]),
            xquat=jp.array(c["xquat"]),
            behavior_labels=jp.array(c["behavior_labels"]),
        )
        for c in clips
    ]

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


# ── Main pipeline ────────────────────────────────────────────────────────────


def _build_step1_generator(config: GenerationConfig):
    """Load Step 1 checkpoint, build env, inference fn, and batched generator."""
    print(f"\nLoading Step 1 checkpoint from {config.checkpoints.step1_path}")
    step1_params = _load_step1_checkpoint(
        config.checkpoints.step1_path,
        config.checkpoints.step1_step,
    )

    cfg = multi_behavior_config()
    cfg.mujoco_impl = "warp"
    gen_env = MultiBehaviorWalker(config=cfg)

    inference_fn = _build_step1_inference_fn(
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

    print(f"JIT compiling Step 1 batched generate (batch_size={batch_n})...")
    np_rng = np.random.RandomState(config.rollout.seed)
    warmup_schedule, _ = _make_transition_schedule(
        np_rng,
        sched.rollout_frames,
        list(range(N_MODES)),
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

    return batched_generate


def run_generation(config: GenerationConfig) -> None:
    """Run the full generation pipeline.

    Dispatches to transitions mode or long_sequence mode based on config.mode.
    """
    if config.mode == "long_sequence":
        _run_long_sequence_generation(config)
    elif config.mode == "transitions":
        _run_transitions_generation(config)
    else:
        raise ValueError(
            f"Unknown generation mode: {config.mode!r}. "
            f"Expected 'transitions' or 'long_sequence'."
        )


def _run_transitions_generation(config: GenerationConfig) -> None:
    total_clips = sum(t.n_clips for t in config.transitions)
    print(f"Walker generation pipeline (transitions mode)")
    print(f"  {len(config.transitions)} transition pairs, {total_clips} total clips")

    batched_generate = _build_step1_generator(config)

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

    activations = _extract_activations(all_clips, config)

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


def _run_long_sequence_generation(config: GenerationConfig) -> None:
    import copy

    ls = config.long_sequence
    print(f"Walker generation pipeline (long_sequence mode)")
    print(f"  {ls.n_clips} clips, {ls.clip_length} frames, {ls.n_transitions} transitions each")

    gen_config = copy.deepcopy(config)
    gen_config.schedule.rollout_frames = max(config.schedule.rollout_frames, ls.clip_length)

    batched_generate = _build_step1_generator(gen_config)

    np_rng = np.random.RandomState(config.rollout.seed)
    jax_rng = jax.random.PRNGKey(config.rollout.seed)

    t_start = time.time()
    all_clips, all_metas, jax_rng = _generate_long_sequence_clips(
        batched_generate, np_rng, jax_rng, gen_config,
    )
    elapsed = time.time() - t_start
    print(f"\nStep 1 done: {len(all_clips)} clips in {elapsed:.1f}s")

    gen_config.schedule.clip_length = ls.clip_length
    activations = _extract_activations(all_clips, gen_config)

    print(f"\nWriting output...")
    write_unified_h5(
        config.output.h5_path,
        latent_mean=activations["latent_mean"],
        agent_qpos=activations["agent_qpos"],
    )

    clip_segment_lists = [m.segments for m in all_metas]
    write_multi_segment_labels_csv(config.output.csv_path, clip_segment_lists)

    if config.output.demo_clip_video and len(all_clips) > 0:
        _render_demo_clip(
            all_clips[0], all_metas[0], config.output.demo_clip_video,
        )

    print(f"\nPipeline complete.")
    print(f"  H5:  {config.output.h5_path}")
    print(f"  CSV: {config.output.csv_path}")


def _print_pair_summary(clip_metas: list[ClipMeta]) -> None:
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for m in clip_metas:
        pair_counts[(m.src_mode, m.dst_mode)] += 1

    print("\nCollected transition pairs:")
    for (src, dst), count in sorted(pair_counts.items()):
        src_name = WALKER_MODES[src]
        dst_name = WALKER_MODES[dst]
        print(f"  {src_name:>15s} -> {dst_name:<15s}: {count}")
    print(f"  Total: {len(clip_metas)} clips")

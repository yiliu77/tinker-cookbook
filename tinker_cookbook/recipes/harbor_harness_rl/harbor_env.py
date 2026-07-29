"""Harbor-harness RL env: rollouts are full Harbor trials whose OpenHands agent
samples through the Tinker proxy. Tokens are read from the proxy captures and
converted to a TrajectoryGroup for the cookbook RL loop.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chz
import tinker
from harbor.models.trial.config import (
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.trial.trial import Trial

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.recipes.harbor_harness_rl.harnesses import HarnessConfig, OpenHandsConfig
from tinker_cookbook.rl.data_processing import _flatten_chunks, _is_prefix
from tinker_cookbook.rl.types import (
    STOP_METRIC_PREFIX,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    StopReason,
    Trajectory,
    TrajectoryGroup,
    Transition,
)

PROXY_IMPORT_PATH = (
    "tinker_cookbook.recipes.harbor_harness_rl.environment.modal:ProxiedModalEnvironment"
)


def _count_prefix_segments(traj: Trajectory) -> int:
    """Number of packed segments (datums) a trajectory compacts into.

    Mirrors the prefix packing in ``trajectory_to_data``: consecutive
    observations that extend the previous ``ob + ac`` merge into one datum; an
    observation that is not a prefix extension starts a new segment. ``1`` means
    the whole rollout packs into a single datum; ``> 1`` signals fragmentation
    (e.g. the harness reformats history or context compaction breaks the prefix).
    """
    acc: list[Any] = []
    segments = 0
    for transition in traj.transitions:
        ob_flat = _flatten_chunks(transition.ob.chunks)
        if not acc:
            acc = list(ob_flat)
        elif _is_prefix(acc, ob_flat):
            acc.extend(ob_flat[len(acc) :])
        else:
            segments += 1
            acc = list(ob_flat)
        acc.extend(transition.ac.tokens)
    if acc:
        segments += 1
    return segments


def _infer_stop_reason(captures: list[dict[str, Any]], max_turns: int, errored: bool) -> str | None:
    """Best-effort trajectory stop reason for a black-box Harbor trial.

    Inferred from the proxy captures (turn count and the last turn's sampler
    finish reason) plus whether the trial raised. Returns ``None`` when the
    reason is unknown (errored or captured no turns) so no ``stop/<reason>``
    metric is emitted.
    """
    if errored or not captures:
        return None
    if len(captures) >= max_turns:
        return StopReason.MAX_TURNS
    if captures[-1]["finish_reason"] == "length":
        return StopReason.MAX_TOKENS
    return StopReason.COMPLETED


def captures_to_trajectory(
    captures: list[dict[str, Any]], stop_reason: str | None = None
) -> Trajectory:
    """One Transition per proxy-captured completion (prompt -> sampled tokens).

    Following the cookbook convention, the final transition carries the one-hot
    ``stop/<reason>`` metric and the trajectory records ``stop_reason``; each
    action also mirrors its per-turn sampler stop reason.
    """
    transitions: list[Transition] = []
    n = len(captures)
    for i, cap in enumerate(captures):
        is_last = i == n - 1
        sampler_stop = "length" if cap["finish_reason"] == "length" else "stop"
        metrics: Metrics = {}
        if is_last and stop_reason is not None:
            metrics[f"{STOP_METRIC_PREFIX}{stop_reason}"] = 1.0
        transitions.append(
            Transition(
                ob=tinker.ModelInput.from_ints(cap["prompt_token_ids"]),
                ac=TokensWithLogprobs(
                    tokens=cap["completion_token_ids"],
                    maybe_logprobs=cap["logprobs"],
                    stop_reason=sampler_stop,
                ),
                reward=0.0,
                episode_done=is_last,
                metrics=metrics,
            )
        )
    return Trajectory(
        transitions=transitions,
        final_ob=tinker.ModelInput.empty(),
        stop_reason=stop_reason,
    )


class HarborHarnessEnvGroupBuilder(EnvGroupBuilder):
    """Runs `group_size` Harbor trials for one task and returns a TrajectoryGroup."""

    def __init__(
        self,
        task_path: str,
        group_size: int,
        max_tokens: int,
        max_trajectory_tokens: int,
        max_turns: int,
        temperature: float,
        agent_timeout_sec: float,
        force_build: bool,
        override_cpus: int,
        override_memory_mb: int,
        override_storage_mb: int,
        harness_config: HarnessConfig,
    ):
        self.task_path = task_path
        self.group_size = group_size
        self.max_tokens = max_tokens
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_turns = max_turns
        self.temperature = temperature
        self.agent_timeout_sec = agent_timeout_sec
        self.force_build = force_build
        self.override_cpus = override_cpus
        self.override_memory_mb = override_memory_mb
        self.override_storage_mb = override_storage_mb
        self.harness_config = harness_config

    async def make_envs(self) -> Sequence[Any]:
        return []

    def logging_tags(self) -> list[str]:
        return [Path(self.task_path).name]

    def _trial_config(self, sampling_client_b64: str) -> TrialConfig:
        agent = self.harness_config.prep_agent(
            max_turns=self.max_turns,
            temperature=self.temperature,
            agent_timeout_sec=self.agent_timeout_sec,
            max_input_tokens=self.max_trajectory_tokens,
            max_output_tokens=self.max_tokens - self.max_trajectory_tokens,
        )
        return TrialConfig(
            task=TaskConfig(path=Path(self.task_path)),
            trial_name=uuid.uuid4().hex[:12],
            agent=agent,
            environment=EnvironmentConfig(
                import_path=PROXY_IMPORT_PATH,
                force_build=self.force_build,
                override_cpus=self.override_cpus,
                override_memory_mb=self.override_memory_mb,
                override_storage_mb=self.override_storage_mb,
                kwargs={
                    "tinker_sampling_client_b64": sampling_client_b64,
                    "max_input_tokens": self.max_trajectory_tokens,
                    "max_tokens": self.max_tokens,
                    "harness_config": chz.asdict(self.harness_config),
                },
            ),
        )

    def _trajectory_metrics(
        self,
        captures: list[dict[str, Any]],
        result: Any,
        errored: bool,
        trajectory: Trajectory,
    ) -> Metrics:
        """Per-trajectory metrics for one trial (turns, token usage, cache, errors)."""
        metrics: Metrics = {"num_turns": float(len(captures))}
        if errored:
            metrics["trial_error"] = 1.0
        n_input, _n_cache, n_output, cost = result.compute_token_cost_totals()
        if n_input is not None:
            metrics["n_input_tokens"] = float(n_input)
        if n_output is not None:
            metrics["n_output_tokens"] = float(n_output)
        if cost is not None:
            metrics["cost_usd"] = float(cost)
        # Fraction of observation (prompt) tokens served from Tinker's prefix cache.
        ob_cache_hit_tokens = sum(cap["prompt_cache_hit_tokens"] for cap in captures)
        ob_tokens = sum(len(cap["prompt_token_ids"]) for cap in captures)
        metrics["ob_cache_hit_tokens"] = float(ob_cache_hit_tokens)
        metrics["ob_cache_hit_frac"] = ob_cache_hit_tokens / ob_tokens if ob_tokens else 0.0
        # Segments the rollout compacts into under trajectory_to_data prefix packing.
        metrics["num_segments"] = float(_count_prefix_segments(trajectory))
        return metrics

    async def _run_trial(self, sampling_client_b64: str) -> tuple[Trajectory, float, Metrics]:
        trial = await Trial.create(self._trial_config(sampling_client_b64))
        result = await trial.run()
        captures = list(getattr(trial.agent_environment, "captured_completions", []))
        rewards = (result.verifier_result.rewards if result.verifier_result else None) or {}
        reward = float(next(iter(rewards.values()), 0.0))
        errored = result.exception_info is not None
        stop_reason = _infer_stop_reason(captures, self.max_turns, errored)
        trajectory = captures_to_trajectory(captures, stop_reason)
        metrics = self._trajectory_metrics(captures, result, errored, trajectory)
        return trajectory, reward, metrics

    async def run_group(self, sampling_client_b64: str) -> TrajectoryGroup:
        results = await asyncio.gather(
            *(self._run_trial(sampling_client_b64) for _ in range(self.group_size))
        )
        return TrajectoryGroup(
            trajectories_G=[traj for traj, _, _ in results],
            final_rewards_G=[reward for _, reward, _ in results],
            metrics_G=[metrics for _, _, metrics in results],
        )


class HarborHarnessDataset(RLDataset):
    def __init__(self, builders: list[HarborHarnessEnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        return self.builders[start : start + self.batch_size]

    def __len__(self) -> int:
        return (len(self.builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class HarborHarnessDatasetBuilder(RLDatasetBuilder):
    task_paths: list[str]
    group_size: int
    groups_per_batch: int
    max_tokens: int = 65536
    max_trajectory_tokens: int = 32 * 1024
    max_turns: int = 20
    temperature: float = 1.0
    agent_timeout_sec: float = 900.0
    force_build: bool = False
    override_cpus: int = 4
    override_memory_mb: int = 5000
    override_storage_mb: int = 5000
    harness_config: HarnessConfig = chz.field(default_factory=OpenHandsConfig)

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        builders = [
            HarborHarnessEnvGroupBuilder(
                task_path=task_path,
                group_size=self.group_size,
                max_tokens=self.max_tokens,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_turns=self.max_turns,
                temperature=self.temperature,
                agent_timeout_sec=self.agent_timeout_sec,
                force_build=self.force_build,
                override_cpus=self.override_cpus,
                override_memory_mb=self.override_memory_mb,
                override_storage_mb=self.override_storage_mb,
                harness_config=self.harness_config,
            )
            for task_path in self.task_paths
        ]
        return HarborHarnessDataset(builders, self.groups_per_batch), None

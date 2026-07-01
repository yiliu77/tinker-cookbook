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
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.trial.trial import Trial

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.recipes.harbor_harness_rl.harnesses import HarnessConfig, OpenHandsConfig
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)

PROXY_IMPORT_PATH = (
    "tinker_cookbook.recipes.harbor_harness_rl.environment.modal:ProxiedModalEnvironment"
)


def captures_to_trajectory(captures: list[dict[str, Any]]) -> Trajectory:
    """One Transition per proxy-captured completion (prompt -> sampled tokens)."""
    transitions: list[Transition] = []
    for i, cap in enumerate(captures):
        transitions.append(
            Transition(
                ob=tinker.ModelInput.from_ints(cap["prompt_token_ids"]),
                ac=TokensWithLogprobs(
                    tokens=cap["completion_token_ids"], maybe_logprobs=cap["logprobs"]
                ),
                reward=0.0,
                episode_done=(i == len(captures) - 1),
            )
        )
    return Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty())


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
        return []  # unused; the custom rollout drives full Harbor trials

    def logging_tags(self) -> list[str]:
        return [Path(self.task_path).name]

    def _trial_config(self, sampling_client_b64: str) -> TrialConfig:
        return TrialConfig(
            task=TaskConfig(path=Path(self.task_path)),
            trial_name=uuid.uuid4().hex[:12],
            agent=AgentConfig(
                name="openhands",
                model_name="hosted_vllm/model",
                env={"LLM_API_KEY": "dummy"},
                # Wall-clock budget for the agent.run() phase (independent of
                # max_iterations); overrides the task's default agent timeout.
                override_timeout_sec=self.agent_timeout_sec,
                model_info={
                    "max_input_tokens": self.max_trajectory_tokens,
                    "max_output_tokens": self.max_tokens - self.max_trajectory_tokens,
                },
                kwargs={
                    "version": "0.60.0 --prerelease=allow",
                    "max_iterations": self.max_turns,
                    "temperature": self.temperature,
                    "num_retries": 1,
                },
            ),
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
                    "harness_config": self.harness_config,
                },
            ),
        )

    async def _run_trial(self, sampling_client_b64: str) -> tuple[list[dict[str, Any]], float]:
        trial = await Trial.create(self._trial_config(sampling_client_b64))
        result = await trial.run()
        captures = list(getattr(trial.agent_environment, "captured_completions", []))
        rewards = (result.verifier_result.rewards if result.verifier_result else None) or {}
        reward = float(next(iter(rewards.values()), 0.0))
        return captures, reward

    async def run_group(self, sampling_client_b64: str) -> TrajectoryGroup:
        results = await asyncio.gather(
            *(self._run_trial(sampling_client_b64) for _ in range(self.group_size))
        )
        return TrajectoryGroup(
            trajectories_G=[captures_to_trajectory(caps) for caps, _ in results],
            final_rewards_G=[reward for _, reward in results],
            metrics_G=[{} for _ in results],
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

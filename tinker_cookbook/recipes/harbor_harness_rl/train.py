"""CLI for Harbor-harness RL training.

Reuses the cookbook RL loop (`train.main` / `do_sync_training`) for weight
saving, advantages, the train step, and logging. The only customization is the
rollout: we pickle the current-weights SamplingClient and ship it to the Harbor
proxy sandbox, run full Harbor trials, and convert the captured tokens into a
TrajectoryGroup.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import pickle
from datetime import datetime
from typing import cast

import chz
import tinker

from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.harbor_harness_rl.harbor_env import (
    HarborHarnessDatasetBuilder,
    HarborHarnessEnvGroupBuilder,
)
from tinker_cookbook.recipes.harbor_harness_rl.harnesses import HarnessConfig, MiniSweAgentConfig
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup
from tinker_cookbook.utils.misc_utils import all_same

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    # Harbor task directories (one EnvGroupBuilder per task).
    task_paths: list[str]

    # Model
    model_name: str = "Qwen/Qwen3.5-4B"
    lora_rank: int = 32

    # Rollout / env
    group_size: int = 1
    groups_per_batch: int = 1

    max_tokens: int = 60000
    max_trajectory_tokens: int = 55000

    # Harnesses treat max_turns differently
    max_turns: int = 120
    temperature: float = 1.0
    agent_timeout_sec: float = 60 * 60  # 1 hour
    force_build: bool = False

    harness_config: HarnessConfig = chz.field(default_factory=MiniSweAgentConfig)

    # Training
    learning_rate: float = 1e-5
    num_substeps: int = 1
    max_steps: int | None = None

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 10
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig) -> None:
    run_name = (
        f"harbor_harness_rl_{cli_config.model_name.replace('/', '-')}"
        f"_gs{cli_config.group_size}_gp{cli_config.groups_per_batch}"
        f"_{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cli_config.log_path or f"/tmp/tinker-examples/harbor_harness_rl/{run_name}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    async def custom_rollout(
        sampling_client: tinker.SamplingClient,
        env_group_builder: EnvGroupBuilder,
        max_tokens: int,
        temperature: float,
        do_remove_constant_reward_groups: bool,
        enable_logging: bool = True,
        strategy=None,
    ) -> TrajectoryGroup | None:
        # Ship the live Tinker sampler to the remote proxy via pickle.
        client_b64 = base64.b64encode(pickle.dumps(sampling_client)).decode()
        builder = cast(HarborHarnessEnvGroupBuilder, env_group_builder)
        tg = await builder.run_group(client_b64)
        if do_remove_constant_reward_groups and all_same(tg.get_total_rewards()):
            return None
        return tg

    # do_sync_training calls this by bare name from the train module namespace.
    train.do_group_rollout_and_filter_constant_reward = custom_rollout

    dataset_builder = HarborHarnessDatasetBuilder(
        task_paths=cli_config.task_paths,
        group_size=cli_config.group_size,
        groups_per_batch=cli_config.groups_per_batch,
        max_tokens=cli_config.max_tokens,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_turns=cli_config.max_turns,
        temperature=cli_config.temperature,
        agent_timeout_sec=cli_config.agent_timeout_sec,
        force_build=cli_config.force_build,
        harness_config=cli_config.harness_config,
    )

    config = train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        recipe_name="recipe_harbor_harness_rl",
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        num_substeps=cli_config.num_substeps,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        log_path=log_path,
        eval_every=0,
        save_every=cli_config.save_every,
        stream_minibatch_config=None,
        max_steps=cli_config.max_steps,
    )

    await train.main(config)


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))

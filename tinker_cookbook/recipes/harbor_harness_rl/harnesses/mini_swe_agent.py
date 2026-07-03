"""mini-swe-agent harness config."""

from __future__ import annotations

import tempfile
from pathlib import Path

import chz
from harbor.models.trial.config import AgentConfig

from tinker_cookbook.recipes.harbor_harness_rl.harnesses.base import (
    HarnessConfig,
    SandboxPrepEnv,
)


@chz.chz
class MiniSweAgentConfig(HarnessConfig):
    """Routes mini-swe-agent's LLM at the proxy and bounds it by step_limit.

    mini-swe-agent runs a ``hosted_vllm/`` model through litellm, which resolves
    its endpoint from ``HOSTED_VLLM_API_BASE``/``HOSTED_VLLM_API_KEY`` in the
    process env (the environment merges ``_persistent_env`` into every command).
    """

    type: str = "mini-swe-agent"
    version: str = ""

    def prep_agent(
        self,
        *,
        max_turns: int,
        temperature: float,
        agent_timeout_sec: float,
        max_input_tokens: int,
        max_output_tokens: int,
    ) -> AgentConfig:
        agent = super().prep_agent(
            max_turns=max_turns,
            temperature=temperature,
            agent_timeout_sec=agent_timeout_sec,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        )
        # step_limit is only settable via a config file; harbor reads this host
        # path at construction, writes it into the container, and passes `-c`.
        config_path = Path(tempfile.gettempdir()) / f"mini_swe_step_limit_{max_turns}.yaml"
        config_path.write_text(f"agent:\n  step_limit: {max_turns}\n")
        agent.kwargs["config_file"] = str(config_path)
        return agent

    async def prep_environment(self, environment: SandboxPrepEnv, proxy_base_url: str) -> None:
        environment._persistent_env["HOSTED_VLLM_API_BASE"] = proxy_base_url
        environment._persistent_env["HOSTED_VLLM_API_KEY"] = "dummy"

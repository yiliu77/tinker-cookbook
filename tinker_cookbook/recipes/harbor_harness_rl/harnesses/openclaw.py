"""OpenClaw harness config."""

from __future__ import annotations

import json

import chz
from harbor.models.trial.config import AgentConfig

from tinker_cookbook.recipes.harbor_harness_rl.harnesses.base import (
    HarnessConfig,
    SandboxPrepEnv,
)

# harbor's OpenClaw copies this container file to ~/.openclaw/openclaw.json at run
# time. On non-mounted envs (Modal) the host-staged copy never reaches the
# container, so we write it here directly.
OPENCLAW_UPLOAD_PATH = "/logs/agent/openclaw.upload.json"


@chz.chz
class OpenClawConfig(HarnessConfig):
    """Routes OpenClaw's OpenAI-compatible LLM at the proxy.

    harbor stages OpenClaw's config through the host log mount, which Modal does
    not provide, so we write the config into the container at the path harbor's
    ``cp`` reads from, pointing the built-in ``openai`` provider at the proxy.
    """

    type: str = "openclaw"
    version: str = ""
    model_name: str = "openai/model"
    thinking: str = "high"

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
        agent.kwargs["thinking"] = self.thinking
        return agent

    async def prep_environment(self, environment: SandboxPrepEnv, proxy_base_url: str) -> None:
        environment._persistent_env["OPENAI_API_KEY"] = "dummy"
        config = {
            "agents": {"defaults": {"workspace": "."}},
            "gateway": {"mode": "local"},
            "tools": {"deny": ["message"]},
            "models": {
                "providers": {
                    "openai": {
                        "baseUrl": proxy_base_url,
                        "api": "openai-completions",
                        "apiKey": "${OPENAI_API_KEY}",
                        "models": [{"id": "model", "name": "model", "reasoning": True}],
                    }
                }
            },
        }
        await environment.write_sandbox_file(OPENCLAW_UPLOAD_PATH, json.dumps(config, indent=2))

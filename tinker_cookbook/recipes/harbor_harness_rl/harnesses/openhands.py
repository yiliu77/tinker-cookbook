"""OpenHands harness config."""

from __future__ import annotations

import chz

from tinker_cookbook.recipes.harbor_harness_rl.harnesses.base import (
    HarnessConfig,
    SandboxPrepEnv,
)


@chz.chz
class OpenHandsConfig(HarnessConfig):
    """Routes OpenHands at the proxy and enables its LLM-summarizing condenser.

    Sets ``LLM_BASE_URL`` for the agent's LLM and drops a ``config.toml`` in the
    working directory (auto-loaded by OpenHands) pointing the condenser LLM at
    the proxy.
    """

    type: str = "openhands"
    version: str = "0.60.0 --prerelease=allow"

    condenser_keep_first: int = 4
    condenser_max_size: int = 80
    condenser_model: str = "hosted_vllm/model"
    workdir: str = "/app"

    async def prep_environment(self, environment: SandboxPrepEnv, proxy_base_url: str) -> None:
        config_toml = (
            "[condenser]\n"
            'type = "llm"\n'
            'llm_config = "condenser"\n'
            f"keep_first = {self.condenser_keep_first}\n"
            f"max_size = {self.condenser_max_size}\n"
            "\n"
            "[llm.condenser]\n"
            f'model = "{self.condenser_model}"\n'
            f'base_url = "{proxy_base_url}"\n'
            'api_key = "dummy"\n'
        )
        await environment.write_sandbox_file(f"{self.workdir}/config.toml", config_toml)

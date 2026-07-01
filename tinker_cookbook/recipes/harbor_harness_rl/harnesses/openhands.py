"""OpenHands harness config."""

from __future__ import annotations

from dataclasses import dataclass

from tinker_cookbook.recipes.harbor_harness_rl.harnesses.base import HarnessConfig


@dataclass
class OpenHandsConfig(HarnessConfig):
    """Enables OpenHands' LLM-summarizing condenser via its config.toml.

    OpenHands auto-loads ``config.toml`` from its working directory, so we drop
    one there pointing the condenser LLM at the proxy.
    """

    condenser_keep_first: int = 4
    condenser_max_size: int = 20
    condenser_model: str = "hosted_vllm/model"
    workdir: str = "/app"

    def sandbox_files(self, proxy_base_url: str) -> dict[str, str]:
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
        return {f"{self.workdir}/config.toml": config_toml}

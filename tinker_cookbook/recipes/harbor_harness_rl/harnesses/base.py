"""Per-harness setup config.

Each harness (OpenHands, mini-swe-agent, ...) needs different setup to route its
LLM through the Tinker proxy. A HarnessConfig is a ``@chz.chz`` config that owns
that setup via two hooks: ``prep_agent`` (host-side, builds the harbor
``AgentConfig``) and ``prep_environment`` (sandbox-side). Concrete harnesses are
selected/overridden through chz (e.g. from the CLI).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol

import chz
from harbor.models.trial.config import AgentConfig


class SandboxPrepEnv(Protocol):
    """The bits of the agent environment a harness needs during ``prep_environment``."""

    _persistent_env: dict[str, str]

    async def write_sandbox_file(self, path: str, content: str) -> None: ...


@chz.chz
class HarnessConfig:
    type: str = ""
    version: str = ""
    model_name: str = "hosted_vllm/model"

    @classmethod
    def from_dict(cls, spec: dict) -> HarnessConfig:
        harness_type = spec["type"]
        for subclass in cls.__subclasses__():
            if subclass().type == harness_type:
                return subclass(**spec)
        raise ValueError(f"Unknown harness type {harness_type!r}")

    def prep_agent(
        self,
        *,
        max_turns: int,
        temperature: float,
        agent_timeout_sec: float,
        max_input_tokens: int,
        max_output_tokens: int,
    ) -> AgentConfig:
        """Build the harbor AgentConfig for this harness (host side)."""
        return AgentConfig(
            name=self.type,
            model_name=self.model_name,
            env={"LLM_API_KEY": "dummy"},
            override_timeout_sec=agent_timeout_sec,
            model_info={
                "max_input_tokens": max_input_tokens,
                "max_output_tokens": max_output_tokens,
            },
            kwargs={
                "version": self.version,
                "max_iterations": max_turns,
                "temperature": temperature,
                "num_retries": 1,
            },
        )

    @abstractmethod
    async def prep_environment(self, environment: SandboxPrepEnv, proxy_base_url: str) -> None:
        """Route the harness LLM at the proxy (sandbox side)."""
        ...

"""Terminus 2 harness config."""

from __future__ import annotations

import chz
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.models.trial.config import AgentConfig

from tinker_cookbook.recipes.harbor_harness_rl.harnesses.base import (
    HarnessConfig,
    SandboxPrepEnv,
)

TERMINUS_IMPORT_PATH = "tinker_cookbook.recipes.harbor_harness_rl.harnesses.terminus:ProxyTerminus2"


class ProxyTerminus2(Terminus2):
    """Terminus 2 that points its host-side LiteLLM at the per-trial proxy.

    The proxy URL only exists once the environment has started, which happens
    after the agent (and its LiteLLM client) are constructed, so ``api_base`` is
    rebound here in ``setup`` before any completion is requested.
    """

    async def setup(self, environment: BaseEnvironment) -> None:
        proxy_base_url = getattr(environment, "proxy_base_url", None)
        if proxy_base_url is not None:
            self._llm._api_base = proxy_base_url
        await super().setup(environment)


@chz.chz
class TerminusConfig(HarnessConfig):
    """Routes Terminus 2's host-side LiteLLM at the proxy.

    Terminus 2 calls the model in-process via LiteLLM rather than from inside the
    sandbox, so it is wired through ``api_base`` (bound in ``ProxyTerminus2.setup``)
    instead of the sandbox-side ``prep_environment`` used by installed harnesses.
    """

    type: str = "terminus-2"
    model_name: str = "hosted_vllm/model"

    def prep_agent(
        self,
        *,
        max_turns: int,
        temperature: float,
        agent_timeout_sec: float,
        max_input_tokens: int,
        max_output_tokens: int,
    ) -> AgentConfig:
        return AgentConfig(
            import_path=TERMINUS_IMPORT_PATH,
            model_name=self.model_name,
            override_timeout_sec=agent_timeout_sec,
            kwargs={
                "max_turns": max_turns,
                "temperature": temperature,
                "parser_name": "json",
                "enable_summarize": True,
                "model_info": {
                    "max_input_tokens": max_input_tokens,
                    "max_output_tokens": max_output_tokens,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
                "llm_kwargs": {"api_key": "dummy"},
            },
        )

    async def prep_environment(self, environment: SandboxPrepEnv, proxy_base_url: str) -> None:
        return None

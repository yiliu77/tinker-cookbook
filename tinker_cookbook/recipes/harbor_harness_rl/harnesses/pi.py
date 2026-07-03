"""Pi harness config."""

from __future__ import annotations

import json

import chz

from tinker_cookbook.recipes.harbor_harness_rl.harnesses.base import (
    HarnessConfig,
    SandboxPrepEnv,
)

# Pi's config dir (default ~/.pi/agent); PI_CODING_AGENT_DIR overrides it verbatim.
PI_AGENT_DIR = "/tmp/pi-agent"


@chz.chz
class PiConfig(HarnessConfig):
    """Routes Pi's OpenAI-compatible LLM at the proxy.

    Pi does not honor ``OPENAI_BASE_URL``; it only redirects a provider via a
    custom-provider ``models.json``. We point ``PI_CODING_AGENT_DIR`` at a
    writable dir and drop a ``models.json`` there overriding the built-in
    ``openai`` provider's ``baseUrl`` and registering the model id.
    """

    type: str = "pi"
    version: str = ""
    model_name: str = "openai/model"

    async def prep_environment(self, environment: SandboxPrepEnv, proxy_base_url: str) -> None:
        environment._persistent_env["OPENAI_API_KEY"] = "dummy"
        environment._persistent_env["PI_CODING_AGENT_DIR"] = PI_AGENT_DIR
        models_json = json.dumps(
            {
                "providers": {
                    "openai": {
                        "baseUrl": proxy_base_url,
                        "api": "openai-completions",
                        "apiKey": "OPENAI_API_KEY",
                        "models": [{"id": "model"}],
                    }
                }
            },
            indent=2,
        )
        await environment.write_sandbox_file(f"{PI_AGENT_DIR}/models.json", models_json)

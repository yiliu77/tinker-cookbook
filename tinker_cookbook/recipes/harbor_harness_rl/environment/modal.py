"""Modal environment that routes agent completions through a Tinker proxy.

A second Modal sandbox runs a FastAPI server that accepts OpenAI-style chat
completion requests, renders them for a Tinker base model, samples via Tinker's
native ``SamplingClient``, and returns an OpenAI ``chat.completion`` response.
Every (request, response) pair — including raw prompt/completion token IDs — is
recorded and pulled back via a control-token-gated ``/__captured__`` endpoint;
``stop`` grabs the captures before the sandbox is torn down.
"""

from __future__ import annotations

import json
import os
import secrets
import shlex
from pathlib import Path
from typing import Any

import httpx
from harbor.environments.modal import ModalEnvironment
from harbor.models.task.config import NetworkMode, NetworkPolicy
from modal import Image, Sandbox, Secret

from tinker_cookbook.recipes.harbor_harness_rl.harnesses import HarnessConfig

PROXY_SB_PORT = 8000
PROXY_DOCKERFILE = Path(__file__).parent / "Dockerfile.proxy"
PROXY_MODULE = "tinker_cookbook.recipes.harbor_harness_rl.environment.proxy_server"


class ProxiedModalEnvironment(ModalEnvironment):
    """Modal environment with a sidecar proxy that samples via Tinker."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proxy_sandbox = None
        self.proxy_url: str | None = None
        self.proxy_base_url: str | None = None
        self.captured_completions: list[dict[str, Any]] = []
        self._control_token = secrets.token_hex(16)

        # Token budgets the proxy uses (output is the generation cap).
        self._max_input_tokens: int = int(self._kwargs.get("max_input_tokens", 32 * 1024))
        self._max_tokens: int = int(self._kwargs.get("max_tokens", 65536))
        self._capture_path: str | None = self._kwargs.get("proxy_capture_path")

        harness_spec = self._kwargs.get("harness_config")
        self._harness_config = HarnessConfig.from_dict(harness_spec) if harness_spec else None

        self._tinker_api_key_env: str = self._kwargs.get("tinker_api_key_env", "TINKER_API_KEY")
        self._tinker_sampling_client_b64: str = self._kwargs["tinker_sampling_client_b64"]

    async def start(self, force_build: bool) -> None:
        self._network_policy = NetworkPolicy(network_mode=NetworkMode.PUBLIC)
        await super().start(force_build)
        await self._start_proxy()
        assert self.proxy_base_url is not None

        self._persistent_env["LLM_BASE_URL"] = self.proxy_base_url
        if self._harness_config is not None:
            await self._harness_config.prep_environment(self, self.proxy_base_url)

    async def write_sandbox_file(self, path: str, content: str) -> None:
        """Write ``content`` to ``path`` inside the agent sandbox (creating dirs)."""
        parent = os.path.dirname(path)
        await self._sdk_exec(
            f"mkdir -p {shlex.quote(parent)} && cat > {shlex.quote(path)} << 'HARNESSCFG'\n"
            f"{content}\nHARNESSCFG"
        )
        self.logger.info("Wrote sandbox file %s", path)

    async def _start_proxy(self):
        api_key = os.environ.get(self._tinker_api_key_env, "")
        assert api_key, f"{self._tinker_api_key_env} is empty; the proxy will fail to reach Tinker."

        secret_env = {
            "PROXY_PORT": str(PROXY_SB_PORT),
            "PROXY_CONTROL_TOKEN": self._control_token,
            "TINKER_API_KEY": api_key,
            "PROXY_TINKER_SAMPLING_CLIENT_B64": self._tinker_sampling_client_b64,
            "PROXY_MAX_INPUT_TOKENS": str(self._max_input_tokens),
            "PROXY_MAX_TOKENS": str(self._max_tokens),
        }

        image = Image.from_dockerfile(PROXY_DOCKERFILE).add_local_python_source("tinker_cookbook")

        self._proxy_sandbox = await Sandbox.create.aio(
            "python3",
            "-u",
            "-m",
            PROXY_MODULE,
            app=self._app,
            image=image,
            encrypted_ports=[PROXY_SB_PORT],
            secrets=[Secret.from_dict(secret_env)],
            timeout=self._sandbox_timeout,
            block_network=False,
        )
        tunnel = (await self._proxy_sandbox.tunnels.aio())[PROXY_SB_PORT]
        self.proxy_url = str(tunnel.url).rstrip("/")
        self.proxy_base_url = self.proxy_url + "/v1"
        self.logger.info("Proxy ready at %s", self.proxy_base_url)

    async def fetch_captured_completions(self) -> list[dict[str, Any]]:
        """Pull stored (request, response) pairs from the proxy while it's alive."""
        if not self.proxy_url:
            return self.captured_completions
        dump_url = self.proxy_url + "/__captured__"
        try:
            async with httpx.AsyncClient(timeout=120.0) as http:
                resp = await http.get(
                    dump_url, headers={"x-proxy-control-token": self._control_token}
                )
                resp.raise_for_status()
                self.captured_completions = resp.json().get("records", [])
        except Exception as exc:
            self.logger.warning("Failed to fetch captured completions: %s", exc)
        return self.captured_completions

    def _write_captured_completions(self) -> None:
        if not self.captured_completions:
            return
        path = (
            Path(self._capture_path)
            if self._capture_path
            else Path(self.trial_paths.trial_dir) / "proxy_completions.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.captured_completions)
        )
        self.logger.info(
            "Wrote %d captured completions to %s", len(self.captured_completions), path
        )

    async def stop(self, delete: bool) -> None:
        # Grab the in-memory capture store before the proxy sandbox is killed.
        if self._proxy_sandbox is not None:
            await self.fetch_captured_completions()
            self._write_captured_completions()
            try:
                await self._proxy_sandbox.terminate.aio()
            except Exception as exc:
                self.logger.warning("Error terminating proxy sandbox: %s", exc)
            self._proxy_sandbox = None

        await super().stop(delete)

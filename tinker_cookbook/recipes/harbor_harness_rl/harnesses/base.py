"""Per-harness setup config.

Each harness (OpenHands, aider, codex, ...) configures things like context
condensation differently. A HarnessConfig declares files to drop into the agent
sandbox before the harness runs
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class HarnessConfig(ABC):
    @abstractmethod
    def sandbox_files(self, proxy_base_url: str) -> dict[str, str]:
        """Files to write into the agent sandbox before the agent runs.

        Returns a mapping of ``{absolute_sandbox_path: file_contents}``. The
        proxy base URL is passed so configs can point the harness's LLM(s) at
        the proxy.
        """
        ...

"""Per-harness setup configs (files dropped into the agent sandbox)."""

from tinker_cookbook.recipes.harbor_harness_rl.harnesses.base import HarnessConfig
from tinker_cookbook.recipes.harbor_harness_rl.harnesses.openhands import OpenHandsConfig

__all__ = ["HarnessConfig", "OpenHandsConfig"]

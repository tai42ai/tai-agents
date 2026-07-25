"""``TAI_AGENTS_*`` limits shared by the agents in this package.

The concrete ``TaiBaseSettings`` subclass self-registers with the settings
registry, so a live-reload reset re-reads the env with no extra wiring. Every
field carries a positive safe default — never hardcoded at a use site, never
unlimited.
"""

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from tai42_kit.settings import TaiBaseSettings, settings_cache


class AgentsLimitsSettings(TaiBaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAI_AGENTS_")

    # Hard ceiling on the voters list a single voting call may fan out to; an
    # over-limit call raises loudly before any voter LLM runs. Must be positive.
    max_voters: int = Field(default=16, gt=0)

    # How many voters run concurrently within one voting call; the rest queue
    # behind a semaphore. Bounds the parallel LLM cost of one call. Must be
    # positive.
    voter_concurrency: int = Field(default=8, gt=0)

    # Bound on the retrieval agent's embedding-dims probe cache (an LRU); the
    # oldest entry is evicted past this. Must be positive.
    embedding_dims_cache_size: int = Field(default=64, gt=0)


@settings_cache
def agents_limits_settings() -> AgentsLimitsSettings:
    return AgentsLimitsSettings()

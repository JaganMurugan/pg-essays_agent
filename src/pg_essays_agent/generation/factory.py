
from pg_essays_agent.config import Settings
from pg_essays_agent.generation.base import GenerationProvider
from pg_essays_agent.generation.gemini import GeminiGenerationProvider


def create_generation_provider(
    settings: Settings,
) -> GenerationProvider:
    """Create the configured LLM generation provider."""

    return GeminiGenerationProvider(settings)

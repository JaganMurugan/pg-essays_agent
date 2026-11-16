
from unittest.mock import patch

from pg_essays_agent.config import Settings
from pg_essays_agent.generation.gemini import GeminiGenerationProvider
from pg_essays_agent.generation.factory import create_generation_provider
from pg_essays_agent.application import create_generation_provider as create_app_provider


def test_factory_creates_gemini_provider():
    settings = Settings(gemini_api_key="test-key")

    provider = create_generation_provider(settings)

    assert isinstance(provider, GeminiGenerationProvider)


@patch("pg_essays_agent.application.build_generation_provider")
def test_application_factory_uses_supplied_settings(mock_factory):
    settings = Settings(gemini_api_key="test-key")
    expected_provider = object()
    mock_factory.return_value = expected_provider

    result = create_app_provider(settings)

    assert result is expected_provider
    mock_factory.assert_called_once_with(settings)


@patch("pg_essays_agent.application.load_settings")
@patch("pg_essays_agent.application.build_generation_provider")
def test_application_factory_loads_settings_when_missing(
    mock_factory,
    mock_load_settings,
):
    settings = Settings(gemini_api_key="test-key")
    expected_provider = object()

    mock_load_settings.return_value = settings
    mock_factory.return_value = expected_provider

    result = create_app_provider()

    assert result is expected_provider
    mock_load_settings.assert_called_once_with()
    mock_factory.assert_called_once_with(settings)

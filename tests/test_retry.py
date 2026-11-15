
from unittest.mock import Mock, call

import pytest

from pg_essays_agent.utils.retry import is_transient, with_retries


@pytest.mark.parametrize(
    "exception",
    [
        KeyError("missing"),
        TypeError("wrong type"),
        ValueError("invalid value"),
        AssertionError("failed assertion"),
    ],
)
def test_permanent_exceptions_are_not_transient(exception):
    assert is_transient(exception) is False


@pytest.mark.parametrize(
    "status",
    [408, 409, 425, 429, 500, 502, 503],
)
def test_retryable_http_statuses(status):
    exception = Exception("HTTP error")
    exception.status_code = status

    assert is_transient(exception) is True


@pytest.mark.parametrize(
    "status",
    [400, 401, 403, 404, 422],
)
def test_non_retryable_http_statuses(status):
    exception = Exception("HTTP error")
    exception.status_code = status

    assert is_transient(exception) is False


def test_exception_without_status_is_transient():
    assert is_transient(ConnectionError("connection reset")) is True


def test_successful_operation_runs_once():
    operation = Mock(return_value="success")

    result = with_retries(operation, "test operation")

    assert result == "success"
    operation.assert_called_once()


def test_transient_failure_is_retried(monkeypatch):
    operation = Mock(
        side_effect=[
            ConnectionError("temporary failure"),
            "success",
        ]
    )

    sleep = Mock()
    monkeypatch.setattr(
        "pg_essays_agent.utils.retry.time.sleep",
        sleep,
    )
    monkeypatch.setattr(
        "pg_essays_agent.utils.retry.random.random",
        lambda: 0.5,
    )

    result = with_retries(operation, "test operation")

    assert result == "success"
    assert operation.call_count == 2
    sleep.assert_called_once_with(1.0)


def test_permanent_failure_is_not_retried(monkeypatch):
    operation = Mock(side_effect=ValueError("invalid input"))
    sleep = Mock()

    monkeypatch.setattr(
        "pg_essays_agent.utils.retry.time.sleep",
        sleep,
    )

    with pytest.raises(ValueError, match="invalid input"):
        with_retries(operation, "test operation")

    operation.assert_called_once()
    sleep.assert_not_called()


def test_exhausted_retries_raise_last_exception(monkeypatch):
    operation = Mock(
        side_effect=ConnectionError("connection unavailable")
    )
    sleep = Mock()

    monkeypatch.setattr(
        "pg_essays_agent.utils.retry.time.sleep",
        sleep,
    )
    monkeypatch.setattr(
        "pg_essays_agent.utils.retry.random.random",
        lambda: 0.5,
    )

    with pytest.raises(ConnectionError, match="connection unavailable"):
        with_retries(operation, "test operation", retries=3)

    assert operation.call_count == 3
    assert sleep.call_count == 2
    assert sleep.call_args_list == [
        call(1.0),
        call(2.0),
    ]


def test_invalid_retry_configuration():
    with pytest.raises(ValueError):
        with_retries(lambda: "success", "test", retries=0)

    with pytest.raises(ValueError):
        with_retries(lambda: "success", "test", base=-1)

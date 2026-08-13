import importlib.metadata
import logging
from unittest import mock

import pytest

from mlflow.store.artifact import databricks_sdk_artifact_repo
from mlflow.store.artifact.databricks_sdk_artifact_repo import (
    DatabricksSdkArtifactRepository,
    _sdk_supports_large_file_uploads,
)


def test_sdk_supports_large_file_uploads_true():
    with mock.patch("importlib.metadata.version", return_value="0.45.0") as mock_version:
        assert _sdk_supports_large_file_uploads() is True
        mock_version.assert_called_once_with("databricks-sdk")


def test_sdk_supports_large_file_uploads_false_for_old_version():
    with mock.patch("importlib.metadata.version", return_value="0.44.0"):
        assert _sdk_supports_large_file_uploads() is False


@pytest.mark.parametrize(
    "side_effect_or_return",
    [
        {"side_effect": importlib.metadata.PackageNotFoundError("databricks-sdk")},
        {"return_value": None},
    ],
)
def test_sdk_supports_large_file_uploads_missing_version_does_not_raise(side_effect_or_return):
    # On Databricks Serverless the vendored databricks-sdk can report no version, which previously
    # crashed `mlflow.log_model()` with `TypeError` from `Version(None)`. It must now degrade to
    # `False` instead.
    with mock.patch("importlib.metadata.version", **side_effect_or_return):
        assert _sdk_supports_large_file_uploads() is False


def test_close_releases_workspace_client_sessions(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://localhost:8080")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token")
    repo = DatabricksSdkArtifactRepository("/Volumes/catalog/schema/volume/path")
    proxy_session = mock.MagicMock()
    repo.wc.files._cached_storage_proxy_session = proxy_session

    with mock.patch.object(repo.wc.api_client._api_client._session, "close") as mock_close:
        repo.close()

    mock_close.assert_called_once_with()
    proxy_session.close.assert_called_once_with()
    with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
        repo.thread_pool.submit(lambda: None)


def test_close_prefers_a_public_workspace_client_close(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://localhost:8080")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token")
    repo = DatabricksSdkArtifactRepository("/Volumes/catalog/schema/volume/path")
    repo.wc.close = mock.MagicMock()

    with mock.patch.object(repo.wc.api_client._api_client._session, "close") as session_close:
        repo.close()

    repo.wc.close.assert_called_once_with()
    session_close.assert_not_called()


def test_missing_close_path_logs_an_error_once(monkeypatch, caplog):
    monkeypatch.setenv("DATABRICKS_HOST", "https://localhost:8080")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token")
    monkeypatch.setattr(
        databricks_sdk_artifact_repo, "_workspace_client_close_path_verified", False
    )
    with (
        mock.patch(
            "mlflow.store.artifact.databricks_sdk_artifact_repo._workspace_client_session",
            return_value=None,
        ) as session_mock,
        caplog.at_level(logging.ERROR, logger=databricks_sdk_artifact_repo.__name__),
    ):
        DatabricksSdkArtifactRepository("/Volumes/catalog/schema/volume/path")
        DatabricksSdkArtifactRepository("/Volumes/catalog/schema/volume/path")

    session_mock.assert_called()
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "databricks-sdk" in errors[0].getMessage()

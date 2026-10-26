"""Tests for immutable demo-model registration and staging gates."""

import hashlib
import json
from pathlib import Path

import pytest

from fraud_scoring.scoring.model_registry import (
    RegistryError,
    promote_to_staging,
    register_demo_model,
    verify_registered,
    verify_staging,
)
from fraud_scoring.scoring.train_demo import train_demo_model


@pytest.fixture(scope="module")
def demo_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    root = tmp_path_factory.mktemp("registry-demo")
    model_path = root / "demo.txt"
    train_demo_model(model_path)
    return model_path


def test_registers_immutable_candidate(
    tmp_path: Path,
    demo_model: Path,
) -> None:
    manifest = register_demo_model(
        tmp_path,
        demo_model,
        "demo-v1",
    )

    stored = tmp_path / "artifacts" / "demo-v1.txt"

    assert stored.is_file()
    assert stored.read_bytes() == demo_model.read_bytes()
    assert manifest["version"] == "demo-v1"
    assert manifest["features"] == [
        "amount",
        "transactions_5m",
    ]
    assert manifest["objective"] == "binary"
    assert manifest["origin"] == "synthetic"
    assert manifest["production_approved"] is False
    assert manifest["sha256"] == hashlib.sha256(
        stored.read_bytes()
    ).hexdigest()

    assert verify_registered(tmp_path, "demo-v1") == manifest


def test_duplicate_registration_is_rejected(
    tmp_path: Path,
    demo_model: Path,
) -> None:
    register_demo_model(tmp_path, demo_model, "demo-v1")

    before = (
        tmp_path / "artifacts" / "demo-v1.txt"
    ).read_bytes()

    with pytest.raises(
        RegistryError,
        match="already exists",
    ):
        register_demo_model(tmp_path, demo_model, "demo-v1")

    assert (
        tmp_path / "artifacts" / "demo-v1.txt"
    ).read_bytes() == before


@pytest.mark.parametrize(
    "version",
    [
        "",
        "../escape",
        "nested/version",
        "none",
    ],
)
def test_invalid_version_is_rejected(
    tmp_path: Path,
    demo_model: Path,
    version: str,
) -> None:
    with pytest.raises(RegistryError, match="Version"):
        register_demo_model(
            tmp_path,
            demo_model,
            version,
        )


def test_missing_source_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RegistryError,
        match="does not exist",
    ):
        register_demo_model(
            tmp_path,
            tmp_path / "missing.txt",
            "demo-v1",
        )


def test_artifact_tampering_is_detected(
    tmp_path: Path,
    demo_model: Path,
) -> None:
    register_demo_model(tmp_path, demo_model, "demo-v1")

    stored = tmp_path / "artifacts" / "demo-v1.txt"

    with stored.open("ab") as stream:
        stream.write(b"\nunauthorized-change")

    with pytest.raises(
        RegistryError,
        match="SHA-256 mismatch",
    ):
        verify_registered(tmp_path, "demo-v1")


def test_manifest_tampering_is_detected(
    tmp_path: Path,
    demo_model: Path,
) -> None:
    register_demo_model(tmp_path, demo_model, "demo-v1")

    manifest_path = (
        tmp_path / "manifests" / "demo-v1.json"
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest["origin"] = "verified_labeled"

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        RegistryError,
        match="metadata: origin",
    ):
        verify_registered(tmp_path, "demo-v1")


def test_promotion_requires_explicit_confirmation(
    tmp_path: Path,
    demo_model: Path,
) -> None:
    register_demo_model(tmp_path, demo_model, "demo-v1")

    with pytest.raises(
        RegistryError,
        match="--confirm",
    ):
        promote_to_staging(
            tmp_path,
            "demo-v1",
            expected_current=None,
            confirm=False,
        )

    assert not (tmp_path / "staging.json").exists()


def test_first_staging_promotion(
    tmp_path: Path,
    demo_model: Path,
) -> None:
    register_demo_model(tmp_path, demo_model, "demo-v1")

    pointer = promote_to_staging(
        tmp_path,
        "demo-v1",
        expected_current=None,
        confirm=True,
    )

    assert pointer["stage"] == "STAGING"
    assert pointer["version"] == "demo-v1"
    assert pointer["production_approved"] is False
    assert verify_staging(tmp_path) == pointer


def test_stale_expected_version_is_rejected(
    tmp_path: Path,
    demo_model: Path,
) -> None:
    register_demo_model(tmp_path, demo_model, "demo-v1")
    register_demo_model(tmp_path, demo_model, "demo-v2")

    promote_to_staging(
        tmp_path,
        "demo-v1",
        expected_current=None,
        confirm=True,
    )

    with pytest.raises(
        RegistryError,
        match="Staging pointer changed",
    ):
        promote_to_staging(
            tmp_path,
            "demo-v2",
            expected_current=None,
            confirm=True,
        )

    assert verify_staging(tmp_path)["version"] == "demo-v1"


def test_staging_switch_requires_matching_current(
    tmp_path: Path,
    demo_model: Path,
) -> None:
    register_demo_model(tmp_path, demo_model, "demo-v1")
    register_demo_model(tmp_path, demo_model, "demo-v2")

    promote_to_staging(
        tmp_path,
        "demo-v1",
        expected_current=None,
        confirm=True,
    )

    pointer = promote_to_staging(
        tmp_path,
        "demo-v2",
        expected_current="demo-v1",
        confirm=True,
    )

    assert pointer["version"] == "demo-v2"
    assert verify_staging(tmp_path)["version"] == "demo-v2"


def test_staging_pointer_tampering_is_detected(
    tmp_path: Path,
    demo_model: Path,
) -> None:
    register_demo_model(tmp_path, demo_model, "demo-v1")

    promote_to_staging(
        tmp_path,
        "demo-v1",
        expected_current=None,
        confirm=True,
    )

    pointer_path = tmp_path / "staging.json"
    pointer = json.loads(
        pointer_path.read_text(encoding="utf-8")
    )
    pointer["sha256"] = "0" * 64

    pointer_path.write_text(
        json.dumps(pointer),
        encoding="utf-8",
    )

    with pytest.raises(
        RegistryError,
        match="Staging digest differs",
    ):
        verify_staging(tmp_path)


def test_existing_writer_lock_rejects_promotion(
    tmp_path: Path,
    demo_model: Path,
) -> None:
    register_demo_model(tmp_path, demo_model, "demo-v1")

    lock_path = tmp_path / ".write.lock"
    lock_path.mkdir()

    try:
        with pytest.raises(
            RegistryError,
            match="writer lock",
        ):
            promote_to_staging(
                tmp_path,
                "demo-v1",
                expected_current=None,
                confirm=True,
            )
    finally:
        lock_path.rmdir()

    assert not (tmp_path / "staging.json").exists()


def test_tampered_model_cannot_be_promoted(
    tmp_path: Path,
    demo_model: Path,
) -> None:
    register_demo_model(tmp_path, demo_model, "demo-v1")

    artifact = tmp_path / "artifacts" / "demo-v1.txt"
    artifact.write_bytes(b"invalid-lightgbm-artifact")

    with pytest.raises(
        RegistryError,
        match="SHA-256 mismatch",
    ):
        promote_to_staging(
            tmp_path,
            "demo-v1",
            expected_current=None,
            confirm=True,
        )

    assert not (tmp_path / "staging.json").exists()

"""Local, synthetic-only LightGBM model registry.

Registered artifacts and manifests are immutable through this API.
STAGING promotion is explicit and does not deploy a model.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from fraud_scoring.scoring.ml import LightGBMScorer


FEATURE_NAMES = ("amount", "transactions_5m")
VERSION_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z"
)


class RegistryError(ValueError):
    """A registration, integrity, or promotion gate failed."""


def _validate_version(version: str) -> str:
    if (
        not isinstance(version, str)
        or VERSION_PATTERN.fullmatch(version) is None
        or version.lower() == "none"
    ):
        raise RegistryError(
            "Version must be 1-64 ASCII letters, digits, "
            "dots, underscores or hyphens; 'none' is reserved"
        )

    return version


def _paths(
    root: Path,
    version: str,
) -> tuple[Path, Path]:
    version = _validate_version(version)
    return (
        root / "artifacts" / f"{version}.txt",
        root / "manifests" / f"{version}.json",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _check_model(path: Path) -> None:
    """Require exact feature schema and valid binary inference."""
    scorer = LightGBMScorer(path)

    if tuple(scorer.feature_names) != FEATURE_NAMES:
        raise RegistryError(
            "Model feature schema does not match the scoring contract"
        )

    score = scorer.score(
        {
            "amount": 42.50,
            "transactions_5m": 3.0,
        }
    )

    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
    ):
        raise RegistryError("Model failed the inference smoke gate")


def _write_json_exclusive(
    path: Path,
    document: dict[str, object],
) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _read_object(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise RegistryError(f"Symbolic links are not allowed: {path}")

    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as exc:
        raise RegistryError(
            f"Cannot read registry document: {path}"
        ) from exc

    if not isinstance(value, dict):
        raise RegistryError("Registry document must be a JSON object")

    return value


@contextmanager
def _writer_lock(root: Path) -> Iterator[None]:
    """Serialize cooperative registry writers using atomic mkdir."""
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".write.lock"

    try:
        lock_path.mkdir()
    except FileExistsError as exc:
        raise RegistryError(
            "Registry writer lock already exists"
        ) from exc

    try:
        yield
    finally:
        lock_path.rmdir()


def register_demo_model(
    root: Path,
    source: Path,
    version: str,
) -> dict[str, object]:
    """Register a synthetic model without replacing existing versions."""
    root = Path(root)
    source = Path(source)
    artifact_path, manifest_path = _paths(root, version)

    if not source.is_file():
        raise RegistryError(f"Source model does not exist: {source}")

    with _writer_lock(root):
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        if artifact_path.exists() or manifest_path.exists():
            raise RegistryError(
                f"Model version already exists: {version}"
            )

        artifact_created = False
        manifest_created = False

        try:
            with source.open("rb") as input_stream:
                with artifact_path.open("xb") as output_stream:
                    artifact_created = True

                    for chunk in iter(
                        lambda: input_stream.read(1024 * 1024),
                        b"",
                    ):
                        output_stream.write(chunk)

            _check_model(artifact_path)

            manifest: dict[str, object] = {
                "version": version,
                "artifact": f"artifacts/{version}.txt",
                "sha256": _sha256(artifact_path),
                "objective": "binary",
                "features": list(FEATURE_NAMES),
                "origin": "synthetic",
                "production_approved": False,
            }

            with manifest_path.open(
                "x",
                encoding="utf-8",
                newline="\n",
            ) as stream:
                manifest_created = True
                json.dump(
                    manifest,
                    stream,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")

            return manifest
        except BaseException:
            if manifest_created:
                manifest_path.unlink(missing_ok=True)
            if artifact_created:
                artifact_path.unlink(missing_ok=True)
            raise


def verify_registered(
    root: Path,
    version: str,
) -> dict[str, object]:
    """Verify artifact integrity and the inference contract."""
    root = Path(root)
    artifact_path, manifest_path = _paths(root, version)
    manifest = _read_object(manifest_path)

    expected_fields = {
        "version": version,
        "artifact": f"artifacts/{version}.txt",
        "objective": "binary",
        "features": list(FEATURE_NAMES),
        "origin": "synthetic",
        "production_approved": False,
    }

    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            raise RegistryError(
                f"Invalid registered model metadata: {field}"
            )

    digest = manifest.get("sha256")

    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise RegistryError("Invalid model SHA-256 metadata")

    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise RegistryError("Registered model artifact is missing or unsafe")

    if _sha256(artifact_path) != digest:
        raise RegistryError("Registered model SHA-256 mismatch")

    _check_model(artifact_path)

    return manifest


def _read_staging_pointer(
    root: Path,
) -> dict[str, object] | None:
    path = root / "staging.json"

    if not path.exists() and not path.is_symlink():
        return None

    pointer = _read_object(path)

    if (
        pointer.get("stage") != "STAGING"
        or pointer.get("production_approved") is not False
    ):
        raise RegistryError("Invalid staging pointer metadata")

    version = pointer.get("version")
    _validate_version(version)

    digest = pointer.get("sha256")

    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise RegistryError("Invalid staging pointer digest")

    return pointer


def promote_to_staging(
    root: Path,
    version: str,
    *,
    expected_current: str | None,
    confirm: bool,
) -> dict[str, object]:
    """Manually update STAGING with integrity and concurrency gates."""
    root = Path(root)
    _validate_version(version)

    if expected_current is not None:
        _validate_version(expected_current)

    if not confirm:
        raise RegistryError(
            "Explicit --confirm is required for staging promotion"
        )

    with _writer_lock(root):
        candidate = verify_registered(root, version)
        current = _read_staging_pointer(root)

        current_version = (
            current["version"] if current is not None else None
        )

        if current_version != expected_current:
            raise RegistryError(
                "Staging pointer changed: expected "
                f"{expected_current!r}, found {current_version!r}"
            )

        pointer: dict[str, object] = {
            "stage": "STAGING",
            "version": version,
            "sha256": candidate["sha256"],
            "production_approved": False,
        }

        temporary_path = root / f".staging-{uuid4().hex}.tmp"

        try:
            _write_json_exclusive(temporary_path, pointer)
            os.replace(temporary_path, root / "staging.json")
        finally:
            temporary_path.unlink(missing_ok=True)

        return pointer


def verify_staging(root: Path) -> dict[str, object]:
    """Verify that STAGING points to a valid, unchanged artifact."""
    root = Path(root)
    pointer = _read_staging_pointer(root)

    if pointer is None:
        raise RegistryError("No staging model has been selected")

    version = pointer["version"]
    manifest = verify_registered(root, version)

    if pointer["sha256"] != manifest["sha256"]:
        raise RegistryError(
            "Staging digest differs from registered artifact"
        )

    return pointer


def main() -> int:
    """Provide explicit registration and staging lifecycle commands."""
    parser = argparse.ArgumentParser(
        description="Synthetic-only local LightGBM model registry"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/model_registry"),
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    register = commands.add_parser("register-demo")
    register.add_argument("--source", type=Path, required=True)
    register.add_argument("--version", required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--version", required=True)

    promote = commands.add_parser("promote-staging")
    promote.add_argument("--version", required=True)
    promote.add_argument(
        "--expected-current",
        required=True,
        help="'none' when no staging version currently exists",
    )
    promote.add_argument(
        "--confirm",
        action="store_true",
    )

    commands.add_parser("verify-staging")

    args = parser.parse_args()

    try:
        if args.command == "register-demo":
            result = register_demo_model(
                args.registry,
                args.source,
                args.version,
            )
        elif args.command == "verify":
            result = verify_registered(
                args.registry,
                args.version,
            )
        elif args.command == "promote-staging":
            expected = (
                None
                if args.expected_current == "none"
                else args.expected_current
            )
            result = promote_to_staging(
                args.registry,
                args.version,
                expected_current=expected,
                confirm=args.confirm,
            )
        else:
            result = verify_staging(args.registry)
    except (RegistryError, OSError, RuntimeError) as exc:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "passed": True,
                **result,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

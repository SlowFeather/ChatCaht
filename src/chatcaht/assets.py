from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class AssetResult:
    component: str
    path: Path
    ok: bool
    detail: str


def verify_manifest(path: str | Path) -> list[AssetResult]:
    manifest_path = Path(path).resolve()
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    root = manifest_path.parent.parent
    results: list[AssetResult] = []
    for component in data.get("components", []):
        name = str(component["name"])
        for asset in component.get("assets", []):
            target = (root / str(asset["path"])).resolve()
            if not target.is_file():
                results.append(AssetResult(name, target, False, "missing"))
                continue
            expected_size = int(asset["size"])
            if target.stat().st_size != expected_size:
                results.append(
                    AssetResult(name, target, False, f"size mismatch: {target.stat().st_size} != {expected_size}")
                )
                continue
            digest = _sha256(target)
            expected_hash = str(asset["sha256"]).lower()
            results.append(
                AssetResult(name, target, digest == expected_hash, "sha256 ok" if digest == expected_hash else "sha256 mismatch")
            )
    return results


def provision_missing(path: str | Path) -> list[AssetResult]:
    manifest_path = Path(path).resolve()
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    root = manifest_path.parent.parent
    failed = {result.component for result in verify_manifest(manifest_path) if not result.ok}
    for component in data.get("components", []):
        name = str(component["name"])
        if name not in failed:
            continue
        provision = component.get("provision") or {}
        command = provision.get("command")
        if not command:
            print(f"[{name}] {provision.get('manual_hint', 'no automatic provision command')}")
            continue
        cwd = (root / str(provision.get("cwd") or ".")).resolve()
        print(f"[{name}] running: {' '.join(str(part) for part in command)}")
        subprocess.run([str(part) for part in command], cwd=cwd, check=True)
    return verify_manifest(manifest_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

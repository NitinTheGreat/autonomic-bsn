"""Provider-dimensioned result paths.

Provider is a DIMENSION of every measurement, not a special case. A number is
meaningless without knowing which stack produced it, so results live under

    results/paper/by_provider/<provider>_<model>/
    results/paper/cross_provider/
    results/paper/tables.md
    results/paper/reproducibility.md

Every result-writing script resolves its folder from the *resolved backend*
rather than hardcoding a path, so switching `LLM_PROVIDER` automatically writes
somewhere else and two providers can never overwrite each other's numbers.
"""

from __future__ import annotations

import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.join(REPO_ROOT, "results", "paper")
BY_PROVIDER = os.path.join(PAPER_DIR, "by_provider")
CROSS_PROVIDER = os.path.join(PAPER_DIR, "cross_provider")


def slug(s: str) -> str:
    """Filesystem-safe fragment, preserving readability (gpt-4o stays gpt-4o)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s or "unknown")).strip("-")


def provider_dir_name(provider: str, model: str) -> str:
    return "%s_%s" % (slug(provider), slug(model))


def provider_dir(provider: str, model: str, create: bool = True) -> str:
    p = os.path.join(BY_PROVIDER, provider_dir_name(provider, model))
    if create:
        os.makedirs(p, exist_ok=True)
    return p


def resolve_from_cfg(cfg: dict, backend: str, create: bool = True) -> str:
    """Folder for the backend as actually resolved, including env overrides."""
    b = (cfg.get("backends") or {}).get(backend, {})
    model = b.get("model", backend)
    env_model = b.get("model_env")
    if env_model:
        model = os.environ.get(env_model, "").strip() or model
    return provider_dir(backend, model, create=create)


def list_provider_dirs() -> list[tuple[str, str, str]]:
    """(provider, model, path) for every provider folder that exists."""
    if not os.path.isdir(BY_PROVIDER):
        return []
    out = []
    for name in sorted(os.listdir(BY_PROVIDER)):
        p = os.path.join(BY_PROVIDER, name)
        if not os.path.isdir(p):
            continue
        provider, _, model = name.partition("_")
        out.append((provider, model, p))
    return out


def write_json(path_dir: str, filename: str, payload: dict) -> str:
    """Write a result file and mirror it for the frontend."""
    import json
    os.makedirs(path_dir, exist_ok=True)
    p = os.path.join(path_dir, filename)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    # Mirror under frontend/ so http.server (which cannot serve above its root)
    # can reach it, preserving the provider dimension in the path.
    rel = os.path.relpath(p, os.path.join(REPO_ROOT, "results"))
    mirror = os.path.join(REPO_ROOT, "frontend", "results", rel)
    os.makedirs(os.path.dirname(mirror), exist_ok=True)
    with open(mirror, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return p

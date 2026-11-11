"""Local profile and display settings for the dashboard.

There are no accounts or logins: the profile is a small JSON file on this computer
(data/user/profile.json). Nothing is sent anywhere. All values are validated on save.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from src.config import project_root

ROLES = ["Student", "Researcher", "Analyst", "Educator", "Other"]
ACCENTS = {"Violet": "#7c5cff", "Teal": "#14b8a6", "Pink": "#ec4899", "Blue": "#3b82f6"}
PLATFORMS = ["All", "Steam", "Nintendo", "Xbox"]
DEFAULTS: dict[str, Any] = {
    "display_name": "Guest",
    "role": "Student",
    "organisation": "",
    "email": "",
    "accent": "Violet",
    "default_platform": "All",
    "default_age_bands": [],
    "show_notes": True,
}
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class ProfileError(ValueError):
    """The profile contains values that cannot be saved."""


def profile_path() -> Path:
    return project_root() / "data" / "user" / "profile.json"


def validate(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return (clean profile, list of problems). Unknown keys are dropped; missing keys take defaults."""
    clean = dict(DEFAULTS)
    errors: list[str] = []

    name = str(raw.get("display_name", DEFAULTS["display_name"])).strip()
    if not name or len(name) > 60 or _CONTROL.search(name):
        errors.append("Name must be 1-60 characters with no control characters.")
    else:
        clean["display_name"] = name

    role = raw.get("role", DEFAULTS["role"])
    if role in ROLES:
        clean["role"] = role
    else:
        errors.append("Choose a role from the list.")

    org = str(raw.get("organisation", "")).strip()
    if len(org) > 80 or _CONTROL.search(org):
        errors.append("Organisation must be at most 80 characters.")
    else:
        clean["organisation"] = org

    email = str(raw.get("email", "")).strip()
    if email and (len(email) > 120 or not _EMAIL.match(email)):
        errors.append("E-mail address does not look valid.")
    else:
        clean["email"] = email

    accent = raw.get("accent", DEFAULTS["accent"])
    if accent in ACCENTS:
        clean["accent"] = accent
    else:
        errors.append("Choose an accent colour from the list.")

    platform = raw.get("default_platform", DEFAULTS["default_platform"])
    if platform in PLATFORMS:
        clean["default_platform"] = platform
    else:
        errors.append("Choose a platform from the list.")

    bands = raw.get("default_age_bands", [])
    if isinstance(bands, list) and len(bands) <= 10 and all(isinstance(b, str) and len(b) <= 12 for b in bands):
        clean["default_age_bands"] = list(bands)
    else:
        errors.append("Age bands are not valid.")

    clean["show_notes"] = bool(raw.get("show_notes", DEFAULTS["show_notes"]))
    return clean, errors


def load_profile(path: Path | None = None) -> dict[str, Any]:
    """Saved profile merged with defaults. A missing or damaged file gives the defaults."""
    p = Path(path) if path else profile_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("not an object")
    except (OSError, ValueError):
        return dict(DEFAULTS)
    clean, errors = validate(raw)
    if errors:   # keep every valid field, fall back to defaults for the rest
        for key in DEFAULTS:
            probe, probs = validate({key: raw.get(key, DEFAULTS[key])})
            if probs:
                clean[key] = DEFAULTS[key]
    return clean


def save_profile(values: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    clean, errors = validate(values)
    if errors:
        raise ProfileError(" ".join(errors))
    p = Path(path) if path else profile_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(clean, fh, indent=2)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return clean


def reset_profile(path: Path | None = None) -> None:
    p = Path(path) if path else profile_path()
    if p.is_file():
        p.unlink()


def initials(name: str) -> str:
    parts = [w for w in re.split(r"\s+", (name or "").strip()) if w]
    if not parts:
        return "G"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()

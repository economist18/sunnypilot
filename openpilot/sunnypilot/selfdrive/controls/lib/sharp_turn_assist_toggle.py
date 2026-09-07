"""Persistent sharp-turn assist toggle with no native or controls imports."""

import os
from pathlib import Path


TOGGLE_FILENAME = "SharpTurnAssist"


def _toggle_path() -> Path:
  # Keep this outside Params' active key directory: manager clears files there
  # that are not present in the prebuilt native parameter registry.
  return Path(os.environ.get("SHARP_TURN_ASSIST_PATH", f"/data/params/{TOGGLE_FILENAME}"))


def sharp_turn_assist_enabled() -> bool:
  try:
    return _toggle_path().read_text().strip() == "1"
  except (OSError, UnicodeError):
    return False


def set_sharp_turn_assist_enabled(enabled: bool) -> None:
  path = _toggle_path()
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path = path.with_name(f".{path.name}.tmp")
  temporary_path.write_text("1" if enabled else "0")
  os.replace(temporary_path, path)

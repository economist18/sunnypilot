"""Bounded low-speed sharp-turn assist helpers."""

import os
from pathlib import Path
import numpy as np

from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan


PREVIEW_EXTRA_SECONDS = 0.12
MAX_ASSIST_SPEED = 13.4  # 30 mph
FULL_ASSIST_SPEED = 8.0  # ~18 mph
MIN_SHARP_CURVATURE = 0.025  # 40 m radius
FULL_SHARP_CURVATURE = 0.08  # 12.5 m radius
MAX_PREVIEW_BLEND = 0.35
OPPOSING_CURVATURE_THRESHOLD = 0.01
SHARP_TURN_CURVATURE_THRESHOLD = 0.035
SHARP_TURN_MIN_INTEGRATOR_SPEED = 2.0
STOCK_MIN_INTEGRATOR_SPEED = 5.0
TOGGLE_FILENAME = "SharpTurnAssist"


def _toggle_path() -> Path:
  # Use a registered param only to discover the active params directory. The
  # custom flag intentionally bypasses the prebuilt native key registry.
  from openpilot.common.params import Params
  return Path(Params().get_param_path("ShowAdvancedControls")).parent / TOGGLE_FILENAME


def sharp_turn_assist_enabled() -> bool:
  try:
    return _toggle_path().read_text().strip() == "1"
  except OSError:
    return False


def set_sharp_turn_assist_enabled(enabled: bool) -> None:
  path = _toggle_path()
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path = path.with_name(f".{path.name}.tmp")
  temporary_path.write_text("1" if enabled else "0")
  os.replace(temporary_path, path)


def get_integrator_min_speed(enabled: bool, desired_curvature: float) -> float:
  if enabled and abs(desired_curvature) >= SHARP_TURN_CURVATURE_THRESHOLD:
    return SHARP_TURN_MIN_INTEGRATOR_SPEED
  return STOCK_MIN_INTEGRATOR_SPEED


def get_sharp_turn_preview_curvature(stock_curvature: float, yaws, yaw_rates, t_idxs,
                                     v_ego: float, lat_delay: float) -> float:
  """Blend a small amount of additional model preview into the stock request.

  The caller still applies openpilot's standard curvature, lateral acceleration,
  and jerk limits after this function. Opposing turn requests are rejected to
  avoid leading an upcoming left/right transition before the stock planner does.
  """
  if v_ego >= MAX_ASSIST_SPEED:
    return float(stock_curvature)

  yaws = np.asarray(yaws, dtype=float)
  yaw_rates = np.asarray(yaw_rates, dtype=float)
  t_idxs = np.asarray(t_idxs, dtype=float)
  if len(t_idxs) < 2 or len(yaws) != len(t_idxs) or len(yaw_rates) != len(t_idxs):
    return float(stock_curvature)
  if not (np.all(np.isfinite(yaws)) and np.all(np.isfinite(yaw_rates)) and np.all(np.isfinite(t_idxs))):
    return float(stock_curvature)

  action_t = max(float(lat_delay), 0.0) + PREVIEW_EXTRA_SECONDS
  preview_curvature = get_curvature_from_plan(yaws, yaw_rates, t_idxs, v_ego, action_t)
  if not np.isfinite(preview_curvature):
    return float(stock_curvature)

  if (abs(stock_curvature) > OPPOSING_CURVATURE_THRESHOLD and
      abs(preview_curvature) > OPPOSING_CURVATURE_THRESHOLD and
      np.sign(stock_curvature) != np.sign(preview_curvature)):
    return float(stock_curvature)

  sharpness = np.interp(max(abs(stock_curvature), abs(preview_curvature)),
                        [MIN_SHARP_CURVATURE, FULL_SHARP_CURVATURE], [0.0, 1.0])
  speed_factor = np.interp(v_ego, [FULL_ASSIST_SPEED, MAX_ASSIST_SPEED], [1.0, 0.0])
  blend = MAX_PREVIEW_BLEND * sharpness * speed_factor
  return float(stock_curvature + blend * (preview_curvature - stock_curvature))


def get_model_sharp_turn_curvature(model_v2, stock_curvature: float, v_ego: float, lat_delay: float) -> float:
  try:
    return get_sharp_turn_preview_curvature(stock_curvature,
                                            model_v2.orientation.z,
                                            model_v2.orientationRate.z,
                                            model_v2.orientation.t,
                                            v_ego, lat_delay)
  except (AttributeError, TypeError, ValueError):
    return float(stock_curvature)

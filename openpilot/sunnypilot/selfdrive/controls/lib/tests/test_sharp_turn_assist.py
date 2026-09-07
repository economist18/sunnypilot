import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib import sharp_turn_assist
from openpilot.sunnypilot.selfdrive.controls.lib.sharp_turn_assist import (
  MAX_ASSIST_SPEED,
  SHARP_TURN_MIN_INTEGRATOR_SPEED,
  STOCK_MIN_INTEGRATOR_SPEED,
  get_integrator_min_speed,
  get_sharp_turn_preview_curvature,
)


def _turn_plan(v_ego: float, curvatures: np.ndarray):
  t_idxs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
  # With zero yaw rate, curv_from_psis returns 2 * yaw / (v * t).
  yaws = curvatures * v_ego * t_idxs / 2.0
  yaw_rates = np.zeros_like(t_idxs)
  return yaws, yaw_rates, t_idxs


def test_preview_leads_same_direction_sharp_turn():
  v_ego = 6.0
  stock_curvature = 0.02
  curvatures = np.array([0.0, 0.01, 0.02, 0.05, 0.08, 0.1])
  yaws, yaw_rates, t_idxs = _turn_plan(v_ego, curvatures)

  assisted = get_sharp_turn_preview_curvature(stock_curvature, yaws, yaw_rates, t_idxs, v_ego, 0.1)

  assert stock_curvature < assisted < 0.1


def test_preview_is_disabled_at_higher_speed():
  stock_curvature = 0.02
  v_ego = MAX_ASSIST_SPEED
  yaws, yaw_rates, t_idxs = _turn_plan(v_ego, np.full(6, 0.1))
  assert get_sharp_turn_preview_curvature(stock_curvature, yaws, yaw_rates, t_idxs, v_ego, 0.1) == stock_curvature


def test_preview_rejects_opposing_turn():
  stock_curvature = 0.05
  v_ego = 6.0
  yaws, yaw_rates, t_idxs = _turn_plan(v_ego, np.full(6, -0.08))
  assert get_sharp_turn_preview_curvature(stock_curvature, yaws, yaw_rates, t_idxs, v_ego, 0.1) == stock_curvature


def test_preview_falls_back_for_invalid_plan():
  stock_curvature = 0.05
  assert get_sharp_turn_preview_curvature(stock_curvature, [0.0], [0.0], [0.0], 6.0, 0.1) == stock_curvature


def test_integrator_speed_only_changes_for_sharp_turns():
  assert get_integrator_min_speed(False, 0.1) == STOCK_MIN_INTEGRATOR_SPEED
  assert get_integrator_min_speed(True, 0.01) == STOCK_MIN_INTEGRATOR_SPEED
  assert get_integrator_min_speed(True, 0.1) == SHARP_TURN_MIN_INTEGRATOR_SPEED


def test_toggle_file_round_trip(tmp_path, monkeypatch):
  toggle_path = tmp_path / "SharpTurnAssist"
  monkeypatch.setattr(sharp_turn_assist, "_toggle_path", lambda: toggle_path)

  assert not sharp_turn_assist.sharp_turn_assist_enabled()
  sharp_turn_assist.set_sharp_turn_assist_enabled(True)
  assert sharp_turn_assist.sharp_turn_assist_enabled()
  sharp_turn_assist.set_sharp_turn_assist_enabled(False)
  assert not sharp_turn_assist.sharp_turn_assist_enabled()

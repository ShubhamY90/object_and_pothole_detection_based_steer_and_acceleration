"""
control/controller.py — Longitudinal and lateral vehicle control.

Improvements over original
--------------------------
Fix 4 — Safe-centre steering
    Instead of reactively swerving away from one object, we compute a
    "safe x" target by nudging the frame centre away from all detected
    threats.  The error between safe_x and the frame centre becomes the
    steering signal.  This is much more stable than instant steer-step jumps.

Fix 6 — Exponential smoothing on both steer and speed
    steer_out = 0.7 * prev_steer + 0.3 * new_steer
    speed_out = 0.7 * prev_speed + 0.3 * desired_speed
    Eliminates jitter and chatter.

Fix 7 — Slow down while turning
    speed_factor *= (1 - abs(steer))
    Sharp turns automatically reduce speed, improving stability.

Priority order (unchanged from original):
  1. Lane-keeping   (road mask centre)
  2. Safe-centre obstacle avoidance
  3. Least-blocked-region fallback (all zones occupied)
"""

from dataclasses import dataclass
from typing import Optional, List

import numpy as np

import config
from filtering.region_filter import Region, FilterResult
from perception.detector import Detection
from perception.road_detector import RoadResult   # RoadDetector output


@dataclass
class VehicleState:
    """Live snapshot of the simulated vehicle."""
    velocity:     float = config.INITIAL_VELOCITY
    steering:     float = 0.0
    acceleration: float = 0.0
    action_label: str   = "GO"
    steer_reason: str   = ""

    def __str__(self) -> str:
        return (
            f"v={self.velocity:5.1f}  "
            f"a={self.acceleration:+6.1f}  "
            f"steer={self.steering:+.2f} ({self.steer_reason})  "
            f"[{self.action_label}]"
        )


class VehicleController:
    """
    Stateful controller. Call step() once per frame.
    """

    def __init__(self, dt: float = config.DT):
        self.dt    = dt
        self.state = VehicleState()

        # Smoothing memory (Fix 6)
        self._prev_steer: float = 0.0
        self._prev_speed: float = config.INITIAL_VELOCITY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        threat:        Optional[Detection],
        filter_result: Optional[FilterResult] = None,
        lane_result    = None,        # RoadResult or LaneResult, both have .lane_centre_x
        frame_width:   int            = config.FRAME_WIDTH,
        frame_height:  int            = config.FRAME_HEIGHT,
        is_spike:      bool           = False,
    ) -> VehicleState:
        """
        Compute one control step.

        Parameters
        ----------
        threat        : primary on-road Detection, or None
        filter_result : full FilterResult (for per-region counts)
        lane_result   : RoadResult / LaneResult (for road centre x)
        frame_width   : needed to compute centre offset fraction
        frame_height  : needed to normalise safe-centre calculation
        is_spike      : True → noisy depth reading, hold previous accel
        """
        # -- Longitudinal -----------------------------------------------
        if threat is None or is_spike:
            acc, label = config.ACC_GO, "GO"
        else:
            acc, label = self._longitudinal(threat)

        # -- Raw steering -----------------------------------------------
        raw_steer, reason = self._lateral(
            threat, filter_result, lane_result, frame_width, frame_height
        )

        # Fix 6 — Smooth steering EMA
        smooth_steer = 0.7 * self._prev_steer + 0.3 * raw_steer
        smooth_steer = float(np.clip(smooth_steer, config.STEER_MIN, config.STEER_MAX))

        # Fix 7 — Reduce speed proportionally while turning
        speed_factor = 1.0 - abs(smooth_steer)   # [0, 1]
        desired_speed = float(np.clip(
            self._prev_speed + acc * self.dt,
            config.MIN_VELOCITY,
            config.MAX_VELOCITY,
        )) * speed_factor

        # Fix 6 — Smooth speed EMA
        smooth_speed = 0.7 * self._prev_speed + 0.3 * desired_speed
        smooth_speed = float(np.clip(smooth_speed, config.MIN_VELOCITY, config.MAX_VELOCITY))

        # -- Update state -----------------------------------------------
        self.state.acceleration = acc
        self.state.action_label = label
        self.state.steering     = smooth_steer
        self.state.steer_reason = reason
        self.state.velocity     = smooth_speed

        # Advance smoothing memory
        self._prev_steer = smooth_steer
        self._prev_speed = smooth_speed

        return self.state

    # ------------------------------------------------------------------
    # Private: longitudinal
    # ------------------------------------------------------------------

    def _longitudinal(self, threat: Detection):
        depth = threat.depth      # now a geometry-based closeness in [0,1]
        if depth > config.DEPTH_HARD_BRAKE:
            acc, label = config.ACC_HARD_BRAKE, "HARD BRAKE"
        elif depth > config.DEPTH_BRAKE:
            acc, label = config.ACC_BRAKE, "BRAKE"
        elif depth > config.DEPTH_SLOW:
            acc, label = config.ACC_SLOW, "SLOW"
        else:
            acc, label = config.ACC_GO, "GO"

        if threat.approaching:
            acc  += config.ACC_APPROACHING_PENALTY
            label = label + " ⚠ APPROACH"

        return acc, label

    # ------------------------------------------------------------------
    # Private: lateral (safe-centre + three-priority steering)
    # ------------------------------------------------------------------

    def _lateral(
        self,
        threat:        Optional[Detection],
        filter_result: Optional[FilterResult],
        lane_result,
        frame_width:   int,
        frame_height:  int,
    ):
        """
        Return (steer_delta, reason_string).

        Priority 1: road-centre keeping (from segmentation mask)
        Priority 2: safe-centre obstacle avoidance
        Priority 3: steer toward least-blocked region (all blocked)
        """

        # -- Priority 1: Road/Lane-centre keeping -----------------------
        lane_delta   = 0.0
        has_lane     = False
        if lane_result is not None and lane_result.lane_centre_x is not None:
            offset = lane_result.lane_centre_x - (frame_width / 2.0)
            norm_offset = offset / (frame_width / 2.0)
            lane_delta  = float(np.clip(
                norm_offset * config.LANE_CENTRE_STEER_GAIN,
                config.STEER_MIN,
                config.STEER_MAX,
            ))
            has_lane = abs(lane_delta) > 0.03

        # -- Priority 2: Safe-centre avoidance (Fix 4) ------------------
        obstacle_delta = 0.0
        reason         = "cruise"

        if filter_result is not None and filter_result.relevant:
            # All three regions blocked? → Priority 3
            counts      = filter_result.counts
            all_blocked = all(counts.get(r, 0) > 0 for r in Region)
            if all_blocked and filter_result.least_blocked is not None:
                lb_delta = self._steer_toward(filter_result.least_blocked)
                if has_lane:
                    combined = lane_delta * 0.5 + lb_delta * 0.5
                    return float(np.clip(combined, config.STEER_MIN, config.STEER_MAX)), "lane+least-blocked"
                return float(np.clip(lb_delta, config.STEER_MIN, config.STEER_MAX)), "least-blocked"

            # Safe-centre: nudge away from each detected threat
            safe_x = frame_width / 2.0
            for obj in filter_result.relevant:
                obj_cx = (obj.x1 + obj.x2) / 2.0
                if obj_cx < safe_x:
                    safe_x += 40.0   # nudge right
                else:
                    safe_x -= 40.0   # nudge left

            safe_x = float(np.clip(safe_x, 0, frame_width))
            error  = safe_x - (frame_width / 2.0)
            obstacle_delta = float(np.clip(
                0.005 * error,
                config.STEER_MIN,
                config.STEER_MAX,
            ))
            reason = "safe-centre"

        # -- Blend lane + obstacle --------------------------------------
        if has_lane:
            # Lane dominates, obstacle adds urgency
            combined = lane_delta * 0.6 + obstacle_delta * 0.4
            return float(np.clip(combined, config.STEER_MIN, config.STEER_MAX)), "lane+avoid"

        if abs(obstacle_delta) > 1e-6:
            return float(np.clip(obstacle_delta, config.STEER_MIN, config.STEER_MAX)), reason

        return 0.0, "cruise"

    @staticmethod
    def _steer_toward(region: Region) -> float:
        """Steer toward a given region (move away from obstacles elsewhere)."""
        if region == Region.LEFT:
            return -config.STEER_STEP   # steer left
        if region == Region.RIGHT:
            return +config.STEER_STEP   # steer right
        return 0.0                      # CENTER clear → go straight



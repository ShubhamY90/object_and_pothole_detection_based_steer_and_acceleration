"""
decision/behavior_planner.py — The discrete decision layer (The "Brain").

This layer separates "Deciding what to do" from "How to do it physically".
It evaluates threats, filtering results, object trajectories, and pothole detections
to output a high-level BehaviorDirective (target velocity and target steering).
"""

from dataclasses import dataclass
from typing import Optional
import config


@dataclass
class BehaviorDirective:
    """The commanded intent from the Brain to the Execution layer."""
    target_velocity: float
    target_steering: float
    braking:         bool
    mode:            str


class BehaviorPlanner:
    """
    Finite State Machine behavior planner. Uses a hierarchical approach:
    1. Emergency interventions (Pothole avoidance, Spike/Collision)
    2. Obstacle navigation (Follow, Steer Around)
    3. Default behavior (Lane cruise)
    """

    def __init__(self):
        pass

    def plan(
        self,
        threat,
        filter_result,
        lane_result,
        pothole_detected: bool,
        frame_width:  int,
        frame_height: int,
        is_spike:     bool = False,
    ) -> BehaviorDirective:
        """
        Evaluate perception signals and output target dynamics.
        """
        max_v  = getattr(config, "MAX_VELOCITY", 80.0)
        max_st = getattr(config, "MAX_STEERING",  1.0)

        braking  = False
        mode     = "cruise"
        target_v = max_v
        
        # Base lane steering if available
        base_steering = self._calculate_lane_steering(lane_result, frame_width)

        # ----------------------------------------------------------------------
        # 1. Critical Emergency Responders
        # ----------------------------------------------------------------------
        if pothole_detected:
            # Pothole in the trajectory!
            target_v = getattr(config, "POTHOLE_SLOW_SPEED", 20.0)
            target_s = max(min(base_steering + getattr(config, "POTHOLE_STEER_OFFSET", 0.3), max_st), -max_st)
            return BehaviorDirective(target_v, target_s, True, "POTHOLE_AVOID")

        if threat is not None:
            proximity = min(threat.depth / frame_height, 1.0)  # 0=very close

            # Collision imminent OR sudden spike
            if is_spike or proximity < 0.25:
                # Emergency Brake
                target_v = 0.0
                braking  = True
                mode     = "brake"
                # If we have a least blocked side, try to swerve
                target_s = self._calculate_swerve(filter_result, base_steering, max_st)

                return BehaviorDirective(target_v, target_s, braking, mode)
            
            # ------------------------------------------------------------------
            # 2. Obstacle / Following Logic
            # ------------------------------------------------------------------
            # Scale speed inversely with proximity
            target_v = max_v * max(proximity - 0.1, 0.0)
            mode     = "follow"
            
            # Point steering at the object's center to follow it (or stay in lane if preferred)
            # We use an obstacle-adjusted steer here
            cx        = (threat.x1 + threat.x2) / 2.0
            target_s  = (cx / frame_width - 0.5) * 2.0
            target_s  = max(-max_st, min(max_st, target_s))

            return BehaviorDirective(target_v, target_s, braking, mode)

        # ----------------------------------------------------------------------
        # 3. Default Cruising
        # ----------------------------------------------------------------------
        target_v = max_v
        mode     = "cruise"
        target_s = base_steering
        
        return BehaviorDirective(target_v, target_s, braking, mode)

    # ── Private Helpers ────────────────────────────────────────────────────

    def _calculate_lane_steering(self, lane_result, frame_width: int) -> float:
        """Returns standard lane-centering offset in [-1.0, 1.0]."""
        if lane_result is not None and getattr(lane_result, "lane_centre_x", None) is not None:
            offset      = lane_result.lane_centre_x - (frame_width / 2.0)
            norm_offset = offset / (frame_width / 2.0)
            gain        = getattr(config, "LANE_CENTRE_STEER_GAIN", 1.0)
            max_st      = getattr(config, "MAX_STEERING", 1.0)
            return max(-max_st, min(max_st, norm_offset * gain))
        return 0.0

    def _calculate_swerve(self, filter_result, base_steering: float, max_st: float) -> float:
        """Calculates steering adjustment if trying to steer into clear regions."""
        if filter_result is not None and filter_result.least_blocked is not None:
            region_str = str(filter_result.least_blocked)
            swerve_mag = getattr(config, "STEER_STEP", 0.3)
            
            if "LEFT" in region_str:
                return max(-max_st, base_steering - swerve_mag)
            elif "RIGHT" in region_str:
                return min(max_st, base_steering + swerve_mag)
            
        return base_steering

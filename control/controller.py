"""
control/controller.py — Temporally-consistent vehicle controller.

Key improvements over the original:
────────────────────────────────────────────────────────────────────────────
1. VELOCITY EMA  — velocity glides toward its target over multiple frames
   instead of jumping. Alpha (VELOCITY_ALPHA) is read from config.

2. RATE LIMITING — even if the EMA wants a large step, the velocity change
   per frame is capped by MAX_ACCEL_RATE / MAX_DECEL_RATE (km/h · s⁻¹).
   This prevents the sudden +168 km/h·s⁻² spike seen in the first frame.

3. STEERING EMA  — steering is also smoothed via a separate STEERING_ALPHA
   so it doesn't snap frame-to-frame.

4. ACCELERATION SMOOTHING — the *reported* acceleration (dv/dt) is itself
   passed through an EMA (ACCEL_SMOOTH_ALPHA) so the HUD shows a smooth,
   believable number instead of noisy per-frame deltas.

5. TEMPORAL STATE — the controller now remembers its previous velocity,
   steering, and smoothed acceleration instead of recomputing them cold
   every frame.
────────────────────────────────────────────────────────────────────────────
"""

import config


class VehicleState:
    """Snapshot returned by VehicleController.step()."""

    def __init__(
        self,
        velocity:     float = 0.0,
        steering:     float = 0.0,
        acceleration: float = 0.0,
        braking:      bool  = False,
        mode:         str   = "cruise",
    ):
        self.velocity     = velocity
        self.steering     = steering
        self.acceleration = acceleration   # signed: negative = decelerating
        self.braking      = braking
        self.mode         = mode

    def __repr__(self) -> str:
        sign = "+" if self.acceleration >= 0 else ""
        return (
            f"VehicleState(v={self.velocity:+.2f} "
            f"steer={self.steering:+.2f} "
            f"accel={sign}{self.acceleration:.2f} "
            f"brake={self.braking} mode={self.mode})"
        )


class VehicleController:
    """
    Temporally-consistent proportional vehicle controller.

    All smoothing parameters are read from config.py so they can be
    tuned without touching this file.

    Parameters
    ----------
    dt : float
        Time-step in seconds (should match the pipeline frame rate).
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt

        # ── Persistent state (updated every frame) ────────────────────────
        self._velocity:       float = 0.0   # current smoothed velocity
        self._steering:       float = 0.0   # current smoothed steering
        self._smooth_accel:   float = 0.0   # smoothed acceleration (HUD display)
        self._initialized:    bool  = False  # first-frame guard

    # ── Public API ─────────────────────────────────────────────────────────

    def step(
        self,
        directive: "BehaviorDirective",
    ) -> VehicleState:
        """
        Execute the BehaviorDirective using temporal smoothing and physics rate limits.
        """

        # ── Config values (with safe defaults) ───────────────────────────
        max_v            = getattr(config, "MAX_VELOCITY",      80.0)
        max_st           = getattr(config, "MAX_STEERING",       1.0)
        vel_alpha        = getattr(config, "VELOCITY_ALPHA",     0.08)
        steer_alpha      = getattr(config, "STEERING_ALPHA",     0.15)
        accel_sm_alpha   = getattr(config, "ACCEL_SMOOTH_ALPHA", 0.30)
        max_accel_rate   = getattr(config, "MAX_ACCEL_RATE",     8.0)   # km/h / s
        max_decel_rate   = getattr(config, "MAX_DECEL_RATE",    20.0)   # km/h / s

        # ── First-frame bootstrap ─────────────────────────────────────────
        if not self._initialized:
            initial_v = getattr(config, "INITIAL_VELOCITY", 0.0)
            self._velocity      = initial_v
            self._steering      = 0.0
            self._smooth_accel  = 0.0
            self._initialized   = True

        target_v  = directive.target_velocity
        raw_steer = directive.target_steering
        braking   = directive.braking
        mode      = directive.mode

        # ────────────────────────────────────────────────────────────────
        # 1. EMA toward target  (temporal smoothing)
        # ────────────────────────────────────────────────────────────────
        ema_velocity = (1.0 - vel_alpha) * self._velocity + vel_alpha * target_v

        # ────────────────────────────────────────────────────────────────
        # 2. Rate limiting — cap how fast velocity can change per frame
        # ────────────────────────────────────────────────────────────────
        delta = ema_velocity - self._velocity
        max_step_up   =  max_accel_rate * self.dt   # e.g.  8 * 0.1 = 0.8 km/h / frame
        max_step_down = -max_decel_rate * self.dt   # e.g. -20 * 0.1 = -2.0 km/h / frame

        delta = max(max_step_down, min(max_step_up, delta))
        new_velocity = self._velocity + delta
        new_velocity = max(0.0, min(max_v, new_velocity))

        # ────────────────────────────────────────────────────────────────
        # 3. Raw acceleration = dv / dt
        # ────────────────────────────────────────────────────────────────
        raw_accel = (new_velocity - self._velocity) / self.dt

        # ────────────────────────────────────────────────────────────────
        # 4. Smooth the reported acceleration (avoids noisy HUD display)
        # ────────────────────────────────────────────────────────────────
        smooth_accel = (
            (1.0 - accel_sm_alpha) * self._smooth_accel
            + accel_sm_alpha * raw_accel
        )

        # ────────────────────────────────────────────────────────────────
        # 5. Steering with EMA  (temporal smoothing)
        # ────────────────────────────────────────────────────────────────
        raw_steer    = max(-max_st, min(max_st, raw_steer))
        new_steering = (1.0 - steer_alpha) * self._steering + steer_alpha * raw_steer
        new_steering = max(-max_st, min(max_st, new_steering))

        # ────────────────────────────────────────────────────────────────
        # 6. Persist state for the next frame
        # ────────────────────────────────────────────────────────────────
        self._velocity      = new_velocity
        self._steering      = new_steering
        self._smooth_accel  = smooth_accel

        return VehicleState(
            velocity=new_velocity,
            steering=new_steering,
            acceleration=smooth_accel,
            braking=braking,
            mode=mode,
        )
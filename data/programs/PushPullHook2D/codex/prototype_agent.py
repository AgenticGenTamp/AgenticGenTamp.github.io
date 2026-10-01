"""Independent prototype controller for PushPullHook2D (not the submitted approach)."""
import numpy as np


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.low = np.asarray(action_space.low, dtype=np.float32)
        self.high = np.asarray(action_space.high, dtype=np.float32)

    def reset(self, state, info):
        s = np.asarray(state)
        u = np.array([np.cos(s[11]), np.sin(s[11])])
        n = np.array([u[1], -u[0]])  # direction of the hook's short leg
        tip = s[9:11] - s[18] * u
        contact = tip + 0.30 * u
        # Stay on the robot's initial side of the leg so the approach never
        # attempts to cross the long tip (which nearly touches the lower wall).
        side = 1. if np.dot(s[:2]-contact, n) >= 0 else -1.
        approach_n = side*n
        # At full arm extension the suction face meets the long leg at ~0.26.
        grasp_dist = .26 if side > 0 else .21
        self.grasp_base = contact + grasp_dist * approach_n
        self.stage_base = self.grasp_base + 0.25 * approach_n
        self.heading = np.arctan2(-approach_n[1], -approach_n[0])
        self.phase = 0
        self.age = 0
        self.pose_target = None
        self.push_start = None
        init_goal=s[29:31]-s[20:22]
        init_goal=init_goal/max(np.linalg.norm(init_goal),1e-6)
        alignment = float(np.dot(n, init_goal))
        sweep_geometry = alignment < -.90 or (.40 < alignment < .70)
        self.high_down = s[21] > 1.82 and init_goal[1] < -.64 and sweep_geometry
        self.use_elbow = np.dot(n,init_goal)<-.35
        self.dynamic_tool = init_goal[1] > 0
        self.fallback_checked = False
        self.special_attempted = False
        self.best_distance = float(np.linalg.norm(s[29:31]-s[20:22]))
        self.stale = 0

    def _act(self, s, pos=None, theta=None, arm=.1, vac=0.):
        a = np.zeros(5, dtype=np.float32)
        if pos is not None:
            a[:2] = np.clip(np.asarray(pos)-s[:2], -.05, .05)
        if theta is not None:
            a[2] = np.clip(_wrap(theta-s[2]), -.1963495, .1963495)
        a[3] = np.clip(arm-s[4], -.1, .1)
        a[4] = vac
        return np.clip(a, self.low, self.high)

    def _next(self):
        self.phase += 1; self.age = 0; self.pose_target = None

    def get_action(self, state):
        s = np.asarray(state); self.age += 1
        # Contact colors update one frame before the server declares success.
        # Hold the achieved configuration instead of immediately beginning the
        # next closed-loop nudge and pulling the button away again.
        if s[25] > .7 and s[34] > .7:
            return self._act(s, arm=s[4], vac=1.)
        distance = float(np.linalg.norm(s[29:31]-s[20:22]))
        if distance < self.best_distance-.002:
            self.best_distance, self.stale = distance, 0
        elif self.phase >= 6:
            self.stale += 1
        if self.phase in (7, 8) and self.stale > 180 and not self.special_attempted:
            self.phase, self.age, self.pose_target = 20, 0, None
            self.original_hook_theta = float(s[11])
            self.prev_turn_hook, self.turn_stall = float(s[11]), 0
            self.special_attempted = True
        # Retract and orient while safely away from the hook.
        if self.phase == 0:
            if self.age > 4: self._next()
            return self._act(s, theta=self.heading, arm=.1)
        # Descend below the hook before lateral transit.
        if self.phase == 1:
            p = np.array([s[0], .16])
            if abs(s[1]-.16)<.012 or self.age>30: self._next()
            return self._act(s, pos=p, theta=self.heading, arm=.1)
        # Move horizontally to the unobstructed side of it.
        if self.phase == 2:
            p = np.array([self.stage_base[0], .16])
            if abs(s[0]-p[0])<.012 or self.age>70: self._next()
            return self._act(s, pos=p, theta=self.heading, arm=.1)
        # Rise on the safe side of the long leg.
        if self.phase == 3:
            if abs(s[1]-self.stage_base[1])<.015 or self.age>40: self._next()
            return self._act(s, pos=self.stage_base, theta=self.heading, arm=.1)
        # Approach normally, with the suction face aimed at the hook.
        if self.phase == 4:
            if np.linalg.norm(s[:2]-self.grasp_base)<.015 or self.age>20: self._next()
            return self._act(s, pos=self.grasp_base, theta=self.heading, arm=.1)
        # Extend through the long leg and attach.
        if self.phase == 5:
            if self.age>5: self._next()
            return self._act(s, pos=self.grasp_base, theta=self.heading, arm=.2, vac=1.)

        # Keep recomputing short endpoint from observed hook pose.
        u = np.array([np.cos(s[11]), np.sin(s[11])])
        n = np.array([u[1], -u[0]])
        short_tip = s[9:11] + s[19]*n
        button=s[20:22]; target=s[29:31]
        goal=target-button; gd=np.linalg.norm(goal)
        g=goal/max(gd,1e-6)
        perp=np.array([-g[1],g[0]])
        # Prefer the below-button clearance side; upward doglegs can strand the
        # robot against the divider on high layouts.
        if perp[1] > 0:
            perp = -perp
        behind=button-.10*g
        # When the short arm points strongly opposite the desired motion its
        # distal endpoint simply recedes; use the exposed elbow corner instead.
        use_elbow = (np.dot(n,g)<-.35) if self.dynamic_tool else self.use_elbow
        toolpoint = s[9:11] if use_elbow else short_tip

        # If placing that corner behind a high button would put the robot through
        # the divider, rotate the held hook first.  The long leg remains upward,
        # while the short leg becomes a reachable flat pusher normal to the goal.
        if self.phase == 6 and not self.fallback_checked:
            self.fallback_checked = True
            predicted_base = s[:2] + (button-.10*g) - toolpoint
            alternate = short_tip if use_elbow else s[9:11]
            alternate_base = s[:2] + (button-.10*g) - alternate
            hard_out = (predicted_base[1] > 1.20 or predicted_base[1] < .10 or
                        predicted_base[0] > 3.40 or predicted_base[0] < .10)
            alt_ok = (.12 < alternate_base[1] < 1.12 and
                      .12 < alternate_base[0] < 3.38)
            if hard_out and alt_ok and abs(g[1]) < .90 and button[1] > 1.33:
                self.use_elbow = not use_elbow
                self.dynamic_tool = False
                toolpoint = alternate
                predicted_base = alternate_base
            if (button[1] > 1.33 and
                    (predicted_base[1] > 1.08 or predicted_base[1] < .13) and
                    button[1] < 1.90):
                self.phase, self.age, self.pose_target = 20, 0, None
                self.original_hook_theta = float(s[11])
                self.prev_turn_hook, self.turn_stall = float(s[11]), 0
                self.special_attempted = True

        if self.phase == 20:
            desired_u = g if g[1] >= 0 else -g
            desired_hook = np.arctan2(desired_u[1], desired_u[0])
            turn = _wrap(desired_hook-s[11])
            if abs(_wrap(s[11]-self.prev_turn_hook)) < .004:
                self.turn_stall += 1
            else:
                self.turn_stall = 0
            self.prev_turn_hook = float(s[11])
            # Some layouts pin the hook against the divider and forbid this
            # rotation; immediately fall back to the proven corner strategy.
            if self.turn_stall > 8 and abs(turn) > .05:
                self.use_elbow = not self.use_elbow
                self.dynamic_tool = False
                self.phase, self.age, self.pose_target = 6, 0, None
                return self._act(s, theta=s[2], arm=.2, vac=1.)
            if abs(turn)<.025 or self.age>70:
                mid = s[9:11] + .5*s[19]*n
                predicted = s[:2] + (button-.085*g) - mid
                if predicted[1] > 1.08 or predicted[1] < .13:
                    self.phase, self.age = 24, 0
                else:
                    self.phase, self.age, self.pose_target = 21, 0, None
            return self._act(s, theta=s[2]+turn, arm=.2, vac=1.)

        if self.phase == 24:
            turn = _wrap(self.original_hook_theta-s[11])
            if abs(turn)<.025 or self.age>70:
                self.phase, self.age, self.pose_target = 6, 0, None
            return self._act(s, theta=s[2]+turn, arm=.2, vac=1.)

        if self.phase in (21, 22, 23):
            mid = s[9:11] + .5*s[19]*n
            behind_mid = button-.085*g
            if self.phase == 21:
                clear = behind_mid + .14*np.array([-g[1], g[0]])
                if np.linalg.norm(mid-clear)<.025 or self.age>70:
                    self.phase, self.age, self.pose_target = 22, 0, None
                return self._act(s, pos=s[:2]+clear-mid, theta=s[2], arm=.2, vac=1.)
            if self.phase == 22:
                if np.linalg.norm(mid-behind_mid)<.018 or self.age>50:
                    self.phase, self.age = 23, 0
                    self.push_start = button.copy()
                return self._act(s, pos=s[:2]+behind_mid-mid, theta=s[2], arm=.2, vac=1.)
            if self.push_start is not None and np.linalg.norm(button-self.push_start)>.002:
                self.phase, self.age = 22, 0
                return self._act(s, pos=s[:2]+behind_mid-mid, theta=s[2], arm=.2, vac=1.)
            step = .006*g
            if self.age>8:
                self.phase, self.age = 22, 0
            return self._act(s, pos=s[:2]+step, theta=s[2], arm=.2, vac=1.)

        # Move endpoint to a clearance waypoint beside the intended behind point.
        if self.phase == 6:
            clear=behind+.16*perp
            if self.pose_target is None: self.pose_target=s[:2]+clear-toolpoint
            if np.linalg.norm(toolpoint-clear)<.025 or self.age>80: self._next()
            return self._act(s,pos=self.pose_target,theta=self.heading,arm=.2,vac=1.)
        # Slide endpoint directly behind the button.
        if self.phase == 7:
            if self.pose_target is None: self.pose_target=s[:2]+behind-toolpoint
            if np.linalg.norm(toolpoint-behind)<.018 or self.age>40:
                self._next(); self.push_start=button.copy()
            return self._act(s,pos=self.pose_target,theta=self.heading,arm=.2,vac=1.)
        # Make a short push, then reset behind the button after each observed move.
        # Endpoint contacts deflect somewhat according to hook orientation, so these
        # repeated closed-loop nudges are substantially more reliable than one sweep.
        if self.high_down:
            return self._act(s, pos=s[:2]+.006*g, theta=self.heading, arm=.2, vac=1.)
        if self.push_start is not None and np.linalg.norm(button-self.push_start)>.002:
            self.phase=7; self.age=0; self.pose_target=None
            return self._act(s,pos=s[:2]+behind-toolpoint,theta=self.heading,arm=.2,vac=1.)
        desired=button+.04*g
        if self.pose_target is None: self.pose_target=s[:2]+desired-toolpoint
        if self.age>6:
            self.phase=7; self.age=0; self.pose_target=None
        return self._act(s,pos=self.pose_target,theta=self.heading,arm=.2,vac=1.)

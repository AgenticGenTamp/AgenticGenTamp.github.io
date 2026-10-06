import numpy as np

BASE_R = 0.1
BTN_R = 0.05
STICK_W = 0.05
STICK_H = 1.25
XMIN, XMAX = 0.0, 3.5
YMIN, YMAX = 0.0, 2.5
TABLE_Y = 1.25
RX_LO, RX_HI = BASE_R + 0.004, XMAX - BASE_R - 0.004
RY_LO, RY_HI = BASE_R + 0.004, TABLE_Y - BASE_R - 0.004
MAXD = 0.05
MAXTH = np.pi / 16
GRASP_D = 0.113  # distance from robot center to stick bottom when grasping


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space

    # ---------- state helpers ----------
    def _parse(self, state):
        robot = None
        stick = None
        buttons = []
        for n in state.get_object_names():
            o = state.get_object_from_name(n)
            t = o.type.name
            if t == "crv_robot":
                robot = o
            elif t == "rectangle":
                stick = o
            elif t == "circle":
                buttons.append(o)
        g = state.get
        r = dict(x=g(robot, "x"), y=g(robot, "y"), th=g(robot, "theta"),
                 arm=g(robot, "arm_joint"), vac=g(robot, "vacuum"))
        s = dict(x=g(stick, "x"), y=g(stick, "y"), th=g(stick, "theta"))
        btns = []
        for b in buttons:
            pressed = g(b, "color_g") > 0.5 and g(b, "color_r") < 0.5
            btns.append(dict(name=b.name, x=g(b, "x"), y=g(b, "y"), pressed=pressed))
        return r, s, btns

    def reset(self, state, info):
        self.holding = False
        self.grasp_cmd = False
        self.prev = None
        self.prev_action = None
        self.stuck = 0
        self.t = 0
        r, s, _ = self._parse(state)
        scx = s["x"] + STICK_W / 2
        self.phase = "normal"
        if scx < RX_LO + 0.002 or scx > RX_HI - 0.002:
            self.phase = "side_approach"
            self.side_d = 1.0 if scx < XMAX / 2 else -1.0

    def _act(self, dx, dy, dth, vac):
        a = np.array([np.clip(dx, -MAXD, MAXD), np.clip(dy, -MAXD, MAXD),
                      np.clip(dth, -MAXTH, MAXTH), -0.1, vac], dtype=np.float32)
        a = np.clip(a, self.action_space.low, self.action_space.high)
        self.prev_action = a
        return a

    def get_action(self, state):
        self.t += 1
        r, s, btns = self._parse(state)
        rejected = False
        if self.prev is not None and self.prev_action is not None:
            pr = self.prev
            moved = abs(pr["x"] - r["x"]) + abs(pr["y"] - r["y"]) + abs(wrap(pr["th"] - r["th"])) > 1e-9
            wanted = np.abs(self.prev_action[:3]).sum() > 1e-9
            rejected = wanted and not moved
        self.prev = r

        if self.grasp_cmd and r["vac"] > 0.5:
            self.holding = True
            self.grasp_cmd = False
            self.hold_offx = (s["x"] + STICK_W / 2) - r["x"]
            self.hold_offy = s["y"] - r["y"]

        if self.phase != "normal":
            return self._side_step(r, s)
        if not self.holding:
            return self._grasp_step(r, s, rejected)
        return self._press_step(r, s, btns)

    # ---------- relocation of a stick stuck at a side wall ----------
    def _side_step(self, r, s):
        d = self.side_d
        sb = s["y"]
        if d > 0:
            tx = s["x"] + STICK_W + 0.113
            th = np.pi
        else:
            tx = s["x"] - 0.113
            th = 0.0
        if self.phase == "side_approach":
            if not hasattr(self, "side_ty"):
                self.side_ty = float(min(RY_HI, max(sb + 0.04, r["y"])))
            ty = self.side_ty
            dth = wrap(th - r["th"])
            if abs(dth) > 1e-6 or abs(r["y"] - ty) > 1e-6:
                px = tx + d * 0.15
                return self._act(px - r["x"], ty - r["y"], dth, 0.0)
            if abs(r["x"] - tx) > 1e-6:
                return self._act(tx - r["x"], 0.0, 0.0, 0.0)
            if r["vac"] > 0.5:
                self.phase = "side_carry"
                self.carry_x = tx + d * 0.3
            else:
                return self._act(0.0, 0.0, 0.0, 1.0)
        if self.phase == "side_carry":
            if abs(r["x"] - self.carry_x) > 1e-6:
                return self._act(self.carry_x - r["x"], 0.0, 0.0, 1.0)
            self.phase = "normal"
            self.prev_action = None
            return self._act(0.0, 0.0, 0.0, 0.0)
        return self._act(0.0, 0.0, 0.0, 0.0)

    # ---------- grasping ----------
    def _grasp_step(self, r, s, rejected):
        scx = s["x"] + STICK_W / 2
        sb = s["y"]
        gx = float(np.clip(scx, RX_LO, RX_HI))
        gy = sb - GRASP_D
        dth = wrap(np.pi / 2 - r["th"])
        if rejected:
            dth_cmd = 0.0
        else:
            dth_cmd = dth
        # Determine waypoint
        safe_y = sb - 0.25
        if abs(r["x"] - gx) > 0.005:
            if r["y"] > sb - 0.23:
                tx, ty = r["x"], safe_y
            else:
                tx, ty = gx, r["y"]
        else:
            tx, ty = gx, gy
            if abs(dth) > 1e-6:
                ty = min(gy, sb - 0.23)
        dx = tx - r["x"]
        dy = ty - r["y"]
        if rejected and abs(dth) > 1e-6:
            # likely rotation blocked; move away downward
            dy = -MAXD
        at_goal = abs(r["x"] - gx) < 1e-6 and abs(r["y"] - gy) < 1e-6 and abs(dth) < 1e-6
        if at_goal:
            self.grasp_cmd = True
            return self._act(0.0, 0.0, 0.0, 1.0)
        return self._act(dx, dy, dth_cmd, 0.0)

    # ---------- pressing ----------
    def _targets(self, r, b):
        """Return list of candidate boxes (xlo,xhi,ylo,yhi) for robot center."""
        boxes = []
        m = 0.012
        # stick contact (two boxes whose union lies inside the rounded-rect region)
        ox, oy = self.hold_offx, self.hold_offy
        for half_x, ext_y in ((STICK_W / 2 - 0.004, BTN_R - m), (STICK_W / 2 + BTN_R - m, -0.004)):
            xlo = max(b["x"] - ox - half_x, RX_LO, XMIN + STICK_W / 2 + 0.003 - ox)
            xhi = min(b["x"] - ox + half_x, RX_HI, XMAX - STICK_W / 2 - 0.003 - ox)
            ylo = max(b["y"] - ext_y - oy - STICK_H, RY_LO, YMIN + 0.003 - oy)
            yhi = min(b["y"] + ext_y - oy, RY_HI, YMAX - 0.003 - oy - STICK_H)
            if xlo <= xhi and ylo <= yhi:
                boxes.append((xlo, xhi, ylo, yhi))
        # base contact: approximate as point target clamped
        px = float(np.clip(b["x"], RX_LO, RX_HI))
        py = float(np.clip(b["y"], RY_LO, min(RY_HI, YMAX - 0.003 - self.hold_offy - STICK_H)))
        if np.hypot(px - b["x"], py - b["y"]) < BASE_R + BTN_R - m:
            boxes.append((px, px, py, py))
        return boxes

    def _press_step(self, r, s, btns):
        rem = [b for b in btns if not b["pressed"]]
        if not rem:
            return self._act(0, 0, 0, 1.0)
        best = None
        for b in rem:
            for (xlo, xhi, ylo, yhi) in self._targets(r, b):
                tx = float(np.clip(r["x"], xlo, xhi))
                ty = float(np.clip(r["y"], ylo, yhi))
                d = max(abs(tx - r["x"]), abs(ty - r["y"]))
                if best is None or d < best[0]:
                    best = (d, tx, ty)
        if best is None:
            # unreachable: just go toward the first
            b = rem[0]
            tx, ty = np.clip(b["x"], RX_LO, RX_HI), np.clip(b["y"], RY_LO, RY_HI)
        else:
            _, tx, ty = best
        return self._act(tx - r["x"], ty - r["y"], 0.0, 1.0)

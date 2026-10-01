"""Approach for Obstruction3D: clear obstructions, then place the target block."""
import numpy as np
import robot

JN = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
POSE_F = ['pose_x', 'pose_y', 'pose_z']
HALF_F = ['half_extent_x', 'half_extent_y', 'half_extent_z']

TABLE_Z = 0.075
SAFE_Z = 0.28  # transit ceiling
PLACE_BOX = (0.08, 0.42, -0.35, 0.35)
FX = 0.055          # gripper half-extent along the finger (open/close) axis
FY = 0.026          # gripper half-extent perpendicular to the finger axis
TIP_BELOW = 0.030   # lowest gripper geometry below the tool origin
MAX_OPEN = 0.078    # max object width along the finger axis


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space

    # ---------------- state accessors ----------------
    def _q(self, s):
        r = s.get_object_from_name('robot')
        return np.array([float(s.get(r, f)) for f in JN])

    def _base(self, s):
        r = s.get_object_from_name('robot')
        return np.array([float(s.get(r, f)) for f in ['pos_base_x', 'pos_base_y', 'pos_base_rot']])

    def _holding(self, s):
        r = s.get_object_from_name('robot')
        return float(s.get(r, 'grasp_active')) > 0.5

    def _grasp_tf(self, s):
        r = s.get_object_from_name('robot')
        p = np.array([float(s.get(r, f)) for f in ['grasp_tf_x', 'grasp_tf_y', 'grasp_tf_z']])
        qq = [float(s.get(r, f)) for f in ['grasp_tf_qx', 'grasp_tf_qy', 'grasp_tf_qz', 'grasp_tf_qw']]
        T = np.eye(4)
        T[:3, :3] = robot.quat_to_mat(*qq)
        T[:3, 3] = p
        return T

    def _pos(self, s, name):
        o = s.get_object_from_name(name)
        return np.array([float(s.get(o, f)) for f in POSE_F])

    def _half(self, s, name):
        o = s.get_object_from_name(name)
        return np.array([float(s.get(o, f)) for f in HALF_F])

    def _tool(self, s):
        return robot.base_tf(*self._base(s)) @ robot.fk_tool(self._q(s))

    def _obstruction_names(self, s):
        return sorted([n for n in s.get_object_names() if n.startswith('obstruction')])

    # ---------------- planning ----------------
    def reset(self, state, info):
        self.t = 0
        self.blocked = 0
        self.mdscale = 1.0
        self.prog_p = None
        self.prog_t = 0
        self.cmd_q = None
        self.qhist = []
        self.escape = 0
        self.retry = 0
        self.dest_try = 0
        self.yaw_try = 0
        self.esc_k = 0
        self.stage = 'approach'
        self.stage_steps = 0
        self.path = []
        self.done = False
        self.place_dz = getattr(self, 'place_dz', 0.002)
        self._plan(state)

    def _plan(self, s):
        maxtop = TABLE_Z
        maxhz = 0.02
        for n in s.get_object_names():
            if n == 'robot':
                continue
            p = self._pos(s, n)
            h = self._half(s, n)
            maxtop = max(maxtop, p[2] + h[2])
            maxhz = max(maxhz, h[2])
        self.safe_z = float(min(SAFE_Z, max(0.20, maxtop + 0.042 + 2*maxhz)))
        reg_p = self._pos(s, 'target_region')
        reg_h = self._half(s, 'target_region')
        blk_p = self._pos(s, 'target_block')
        blk_h = self._half(s, 'target_block')
        self.blk_target = np.array([reg_p[0], reg_p[1], reg_p[2] + reg_h[2] + blk_h[2]])
        obstr = self._obstruction_names(s)
        move, stay = [], []
        for n in obstr:
            p = self._pos(s, n)
            h = self._half(s, n)
            if (abs(p[0] - reg_p[0]) < blk_h[0] + h[0] + 0.05 and
                    abs(p[1] - reg_p[1]) < blk_h[1] + h[1] + 0.05):
                move.append(n)
            else:
                stay.append(n)
        occupied = [(reg_p[0], reg_p[1], max(reg_h[0], blk_h[0]), max(reg_h[1], blk_h[1])),
                    (blk_p[0], blk_p[1], blk_h[0], blk_h[1])]
        for n in stay:
            p = self._pos(s, n)
            h = self._half(s, n)
            occupied.append((p[0], p[1], h[0], h[1]))
        # tallest first: its fingers then clear the shorter neighbours
        move.sort(key=lambda n: -(self._pos(s, n)[2] + self._half(s, n)[2]))
        self.tasks = []
        for n in move:
            h = self._half(s, n)
            dests = self._dest_candidates(self._pos(s, n), h, occupied, reg_p, reg_h)
            occupied.append((dests[0][0], dests[0][1], h[0], h[1]))
            self.tasks.append({'obj': n, 'dests': dests, 'block': False})
        self.tasks.append({'obj': 'target_block', 'dests': [self.blk_target[:2].copy()], 'block': True})

    def _dest_candidates(self, cur, ho, occupied, reg_p, reg_h):
        cands = []
        xs = np.arange(PLACE_BOX[0], PLACE_BOX[1] + 1e-9, 0.02)
        ys = np.arange(PLACE_BOX[2], PLACE_BOX[3] + 1e-9, 0.02)
        for x in xs:
            for y in ys:
                if not (abs(x - reg_p[0]) > reg_h[0] + ho[0] + 0.08 or
                        abs(y - reg_p[1]) > reg_h[1] + ho[1] + 0.08):
                    continue
                ok = True
                for (ox, oy, ohx, ohy) in occupied:
                    if not (abs(x - ox) > ohx + ho[0] + 0.07 or abs(y - oy) > ohy + ho[1] + 0.07):
                        ok = False
                        break
                if not ok:
                    continue
                d = (x - cur[0]) ** 2 + (y - cur[1]) ** 2
                cands.append((d, np.array([x, y])))
        cands.sort(key=lambda c: c[0])
        out = [c[1] for c in cands[:60]]
        if not out:
            out = [np.array([0.25, -0.33 if reg_p[1] > 0 else 0.33])]
        picked = [out[0]]
        for c in out[1:]:
            if all(np.linalg.norm(c - p) > 0.09 for p in picked):
                picked.append(c)
            if len(picked) >= 5:
                break
        return picked

    def _grasp_z(self, s, name):
        p = self._pos(s, name)
        h = self._half(s, name)
        return max(p[2] + h[2] + 0.031, p[2] - h[2] + 0.047)

    # -------- yaw choice: keep the gripper footprint clear of neighbours --------
    def _yaw_score(self, s, center_xy, tip_z, half_obj, ignore, yaw):
        c, sn = np.cos(yaw), np.sin(yaw)
        Rt = np.array([[c, sn], [-sn, c]])   # world -> gripper frame
        worst = 1e9
        for n in s.get_object_names():
            if n == 'robot' or n in ignore:
                continue
            p = self._pos(s, n)
            h = self._half(s, n)
            if p[2] + h[2] < tip_z + 0.002:
                continue
            d = Rt @ (p[:2] - center_xy)
            rad = np.hypot(h[0], h[1])
            mx = abs(d[0]) - (FX + rad)
            my = abs(d[1]) - (FY + rad)
            worst = min(worst, max(mx, my))
        return worst

    def _choose_yaw(self, s, name, center_xy, gz, dest_xy=None, k=0):
        h = self._half(s, name)
        tip_z = gz - TIP_BELOW
        cands = []
        for yaw in np.arange(0.0, np.pi - 1e-9, np.pi / 12):
            width = 2 * (h[0] * abs(np.cos(yaw)) + h[1] * abs(np.sin(yaw)))
            if width > MAX_OPEN:
                continue
            sc = self._yaw_score(s, center_xy, tip_z, h, {name}, yaw)
            if dest_xy is not None:
                sc = min(sc, self._yaw_score(s, dest_xy, tip_z, h, {name}, yaw))
            cands.append((sc, yaw))
        if not cands:
            return 0.0
        cands.sort(key=lambda c: -c[0])
        return cands[min(k, len(cands) - 1)][1]


    # -------- horizontal transit shaping (arc around the base) --------
    def _horiz_target(self, s, goal_xy, r_min=0.30):
        T = self._tool(s)
        b = self._base(s)
        bxy = np.array([b[0], b[1]])
        vc = T[:2, 3] - bxy
        vg = np.asarray(goal_xy, float) - bxy
        rc = np.linalg.norm(vc)
        rg = np.linalg.norm(vg)
        thc = np.arctan2(vc[1], vc[0])
        thg = np.arctan2(vg[1], vg[0])
        dth = (thg - thc + np.pi) % (2*np.pi) - np.pi
        if abs(dth) < 0.10:
            return np.asarray(goal_xy, float)
        r_safe = min(0.52, max(rc, rg, r_min))
        if rc < r_safe - 0.015:
            return bxy + r_safe*np.array([np.cos(thc), np.sin(thc)])
        thn = thc + np.clip(dth, -0.45, 0.45)
        return bxy + r_safe*np.array([np.cos(thn), np.sin(thn)])


    def _best_clearance(self, s, name, dest):
        gz = self._grasp_z(s, name)
        p = self._pos(s, name)
        best = -9.9
        h = self._half(s, name)
        for yaw in np.arange(0.0, np.pi - 1e-9, np.pi/12):
            width = 2*(h[0]*abs(np.cos(yaw)) + h[1]*abs(np.sin(yaw)))
            if width > MAX_OPEN:
                continue
            sc = self._yaw_score(s, p[:2], gz - TIP_BELOW, h, {name}, yaw)
            best = max(best, sc)
        return best

    def _select_task(self, s):
        """Greedily pick the most accessible remaining obstruction."""
        if len(self.tasks) < 3:
            return
        best_i, best_sc = 0, None
        for i, t in enumerate(self.tasks):
            if t['block']:
                continue
            sc = self._best_clearance(s, t['obj'], t['dests'][0])
            sc = min(sc, 0.02) + 0.02*(self._pos(s, t['obj'])[2] + self._half(s, t['obj'])[2])
            if best_sc is None or sc > best_sc:
                best_sc, best_i = sc, i
        if best_i != 0:
            self.tasks.insert(0, self.tasks.pop(best_i))

    # ---------------- execution ----------------
    def _near_clutter(self, s, ignore=()):
        T = self._tool(s)
        p = T[:3, 3]
        if p[2] > 0.27:
            return False
        for n in s.get_object_names():
            if n == 'robot' or n in ignore or n == 'target_region':
                continue
            op = self._pos(s, n)
            oh = self._half(s, n)
            if op[2] + oh[2] > p[2] - 0.07 and np.hypot(op[0]-p[0], op[1]-p[1]) < 0.10:
                return True
        return False

    def _servo(self, s, p_des, R_des, max_delta=0.2, ignore=()):
        if self._near_clutter(s, ignore):
            max_delta = min(max_delta, 0.02)
        q = self._q(s)
        B = robot.base_tf(*self._base(s))
        pl = (np.linalg.inv(B) @ np.append(p_des, 1.0))[:3]
        Rl = B[:3, :3].T @ R_des
        dq, ep, ew = robot.ik_step(q, pl, Rl, damp=0.03,
                                   max_delta=max(0.003, max_delta*self.mdscale))
        if not np.all(np.isfinite(dq)):
            dq = np.random.uniform(-0.05, 0.05, 7)
        a = np.zeros(11, dtype=np.float32)
        a[3:10] = dq
        self.cmd_q = q + dq
        return a

    def _at(self, s, p_des, R_des, tol, rtol=0.02):
        T = self._tool(s)
        ep = np.linalg.norm(p_des - T[:3, 3])
        ew = np.linalg.norm(robot.rot_log(R_des @ T[:3, :3].T))
        return ep < tol and ew < rtol

    def get_action(self, s):
        self.t += 1
        q = self._q(s)
        if self.cmd_q is not None and np.max(np.abs(q - self.cmd_q)) > 1e-6:
            self.blocked += 1
            self.mdscale = max(0.05, self.mdscale*0.35)
        else:
            self.blocked = 0
            self.mdscale = min(1.0, self.mdscale*1.3)
            if not self.qhist or np.max(np.abs(q - self.qhist[-1])) > 1e-9:
                self.qhist.append(q.copy())
                if len(self.qhist) > 80:
                    self.qhist.pop(0)
        self.cmd_q = None
        if self.escape > 0:
            if self.blocked == 0 and self.esc_k > 1 and not self._holding(s):
                self.escape = 0
                self._defer_task()
                self._set_stage('approach')
            else:
                return self._do_escape(s, q)
        T0 = self._tool(s)
        if self.prog_p is None or np.linalg.norm(T0[:3, 3] - self.prog_p) > 0.004:
            self.prog_p = T0[:3, 3].copy()
            self.prog_t = self.t
        if self.blocked >= 4 and self.t - self.prog_t > 25:
            self.escape = 10
            self.esc_k = 0
            self.blocked = 0
            self.prog_t = self.t
            self._on_jam()
            return self._do_escape(s, q)
        for _ in range(6):
            out = self._step_machine(s)
            if out is not None:
                return out
        return np.zeros(11, dtype=np.float32)

    def _defer_task(self):
        """Push the current task back for a later retry."""
        if not self.tasks or self.tasks[0]['block'] or len(self.tasks) < 2:
            return
        t = self.tasks.pop(0)
        t['fails'] = t.get('fails', 0) + 1
        self.retry = 0
        self.dest_try = 0
        self.yaw_try = 0
        if t['fails'] <= 6:
            self.tasks.insert(max(0, len(self.tasks) - 1), t)

    def _on_jam(self):
        """Change strategy for the current task after a jam."""
        self.yaw_try += 1
        if self.yaw_try > 3:
            self.yaw_try = 0
            self.dest_try += 1
            self._defer_task()

    def _do_escape(self, s, q):
        """Try a menu of small perturbations to break out of a jam."""
        self.escape -= 1
        k = self.esc_k
        self.esc_k += 1
        a = np.zeros(11, dtype=np.float32)
        if self.escape <= 0:
            self._defer_task()
            self._set_stage('approach')
        if k % 3 == 0 and self._holding(s):
            # drop what we hold; it frees the gripper and we can retry later
            a[10] = 1.0
            self.cmd_q = None
            return a
        T = self._tool(s)
        B = robot.base_tf(*self._base(s))
        mode = k % 6
        if mode in (0, 1, 4):
            dz = 0.02 if mode != 4 else -0.01
            md = {0: 0.01, 1: 0.004, 4: 0.005}[mode]
            tgt = T[:3, 3] + np.array([0.0, 0.0, dz])
            pl = (np.linalg.inv(B) @ np.append(tgt, 1.0))[:3]
            Rl = B[:3, :3].T @ T[:3, :3]
            dq, _, _ = robot.ik_step(q, pl, Rl, damp=0.05, max_delta=md)
        elif mode == 2 and self.qhist:
            dq = np.clip(self.qhist[-1] - q, -0.006, 0.006)
        else:
            mag = 0.01 if mode == 3 else 0.03
            dq = np.random.uniform(-mag, mag, 7)
        if not np.all(np.isfinite(dq)):
            dq = np.zeros(7)
        a[3:10] = dq
        self.cmd_q = q + dq
        return a

    def _replay(self, s, ignore=()):
        """Retrace the recorded approach path (joint space) to back out safely."""
        q = self._q(s)
        while self.path and np.max(np.abs(self.path[-1] - q)) < 1e-6:
            self.path.pop()
        if not self.path:
            return None
        tgt = self.path[-1]
        lim = max(0.004, 0.2*self.mdscale)
        dq = np.clip(tgt - q, -lim, lim)
        a = np.zeros(11, dtype=np.float32)
        a[3:10] = dq
        self.cmd_q = q + dq
        return a

    def _set_stage(self, st):
        self.stage = st
        self.stage_steps = 0
        self.blocked = 0

    def _step_machine(self, s):
        if not self.tasks:
            self._replan_final(s)
            if not self.tasks:
                return np.zeros(11, dtype=np.float32)
        task = self.tasks[0]
        name = task['obj']
        dest = task['dests'][min(self.dest_try, len(task['dests']) - 1)]
        self.stage_steps += 1
        if self.stage_steps > 90:
            self._set_stage('approach')
            self.retry += 1
            self.yaw_try += 1
            if self.retry > 3:
                self.dest_try += 1
                self._defer_task()
            return None

        gz = self._grasp_z(s, name) - 0.005*(self.retry % 3)
        T0 = self._tool(s)
        yaw_cur = np.arctan2(T0[1, 0], T0[0, 0])
        if self.stage in ('approach', 'descend', 'close'):
            p = self._pos(s, name)
            yaw = self._choose_yaw(s, name, p[:2], gz, dest,
                                   self.yaw_try + task.get('fails', 0))
            # the gripper is symmetric: pick the equivalent yaw nearest the current one
            yaw = yaw + np.pi*np.round((yaw_cur - yaw)/np.pi)
            self.cur_yaw = yaw
        else:
            yaw = getattr(self, 'cur_yaw', 0.0)
        R = robot.down_R(yaw)

        if self.stage == 'approach':
            if self._holding(s):
                self._set_stage('lift')
                return None
            if self.stage_steps == 1 and not task['block']:
                self._select_task(s)
                task = self.tasks[0]
                name = task['obj']
                dest = task['dests'][min(self.dest_try, len(task['dests']) - 1)]
                gz = self._grasp_z(s, name)
                p = self._pos(s, name)
                yaw = self._choose_yaw(s, name, p[:2], gz, dest,
                                       self.yaw_try + task.get('fails', 0))
                yaw = yaw + np.pi*np.round((yaw_cur - yaw)/np.pi)
                self.cur_yaw = yaw
                R = robot.down_R(yaw)
            p = self._pos(s, name)
            tgt = np.array([p[0], p[1], self.safe_z])
            if self._at(s, tgt, R, 0.004):
                self.path = []
                self._set_stage('descend')
                return None
            xy = self._horiz_target(s, p[:2])
            return self._servo(s, np.array([xy[0], xy[1], self.safe_z]), R, ignore=(name,))

        if self.stage == 'descend':
            p = self._pos(s, name)
            tgt = np.array([p[0], p[1], gz])
            q = self._q(s)
            if not self.path or np.max(np.abs(self.path[-1] - q)) > 1e-7:
                self.path.append(q.copy())
            if self._at(s, tgt, R, 0.0015) or self.blocked >= 4:
                self._set_stage('close')
                return None
            return self._servo(s, tgt, R, max_delta=0.08, ignore=(name,))

        if self.stage == 'close':
            if self._holding(s):
                self._set_stage('lift')
                return None
            if self.stage_steps > 2:
                self.retry += 1
                self.yaw_try += 1
                self._set_stage('approach')
                if self.retry > 2:
                    self._defer_task()
                return None
            a = np.zeros(11, dtype=np.float32)
            a[10] = -1.0
            return a

        if self.stage == 'lift':
            T = self._tool(s)
            if T[2, 3] > self.safe_z - 0.006:
                self.path = []
                self._set_stage('transit')
                return None
            out = self._replay(s, (name,))
            if out is not None:
                return out
            tgt = np.array([T[0, 3], T[1, 3], self.safe_z])
            return self._servo(s, tgt, R, max_delta=0.06 if T[2, 3] < 0.2 else 0.2, ignore=(name,))

        if self.stage == 'transit':
            tgt = np.array([dest[0], dest[1], self.safe_z])
            if self._at(s, tgt, R, 0.004):
                self.path = []
                self._set_stage('lower')
                return None
            xy = self._horiz_target(s, dest[:2])
            return self._servo(s, np.array([xy[0], xy[1], self.safe_z]), R, ignore=(name,))

        if self.stage == 'lower':
            if not self._holding(s):
                self._set_stage('retreat')
                return None
            G = self._grasp_tf(s)
            h = self._half(s, name)
            zc = (self.blk_target[2] if task['block'] else TABLE_Z + h[2]) + self.place_dz
            Tobj = np.eye(4)
            Tobj[:3, :3] = robot.quat_to_mat(*[float(s.get(s.get_object_from_name(name), f))
                                               for f in ['pose_qx', 'pose_qy', 'pose_qz', 'pose_qw']])
            Tobj[:3, :3] = np.eye(3) if task['block'] else Tobj[:3, :3]
            Tobj[:3, 3] = np.array([dest[0], dest[1], zc])
            Tg = Tobj @ np.linalg.inv(G)
            q = self._q(s)
            if not self.path or np.max(np.abs(self.path[-1] - q)) > 1e-7:
                self.path.append(q.copy())
            if self._at(s, Tg[:3, 3], Tg[:3, :3], 0.0012) or self.blocked >= 6:
                self._set_stage('open')
                return None
            return self._servo(s, Tg[:3, 3], Tg[:3, :3], max_delta=0.06, ignore=(name,))

        if self.stage == 'open':
            if not self._holding(s):
                self.tasks.pop(0)
                self.retry = 0
                self.dest_try = 0
                self.yaw_try = 0
                self._set_stage('retreat')
                return None
            if self.stage_steps > 2:
                self.dest_try += 1
                self.retry += 1
                if self.retry > 4:
                    self.dest_try = 0
                    self.retry = 0
                self._set_stage('lift')
                return None
            a = np.zeros(11, dtype=np.float32)
            a[10] = 1.0
            return a

        if self.stage == 'retreat':
            T = self._tool(s)
            if T[2, 3] > self.safe_z - 0.01 or self.stage_steps > 20:
                self.path = []
                self._set_stage('approach')
                return None
            out = self._replay(s, (name,))
            if out is not None:
                return out
            tgt = np.array([T[0, 3], T[1, 3], self.safe_z])
            return self._servo(s, tgt, R, max_delta=0.06 if T[2, 3] < 0.2 else 0.2, ignore=(name,))

        self._set_stage('approach')
        return None

    def _replan_final(self, s):
        blk_p = self._pos(s, 'target_block')
        self.replans = getattr(self, 'replans', 0) + 1
        if self.replans > 8:
            self.tasks = [{'obj': 'target_block',
                           'dests': [self.blk_target[:2].copy()], 'block': True}]
        else:
            self._plan(s)
        self.retry = 0
        self.dest_try = 0
        self.yaw_try = 0
        self._set_stage('lift' if self._holding(s) else 'approach')

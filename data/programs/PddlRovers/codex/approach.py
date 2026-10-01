"""Geometry-aware policy for the object-centric Rovers benchmark."""

import heapq
import math

import numpy as np


SAMPLE, CALIBRATE, IMAGE, NOOP, SEND, DROP = (
    -5.0 / 6.0, -0.5, -1.0 / 6.0, 1.0 / 6.0, 0.5, 5.0 / 6.0
)


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.space = observation_space
        self.types = {str(t): t for t in observation_space.types}
        # Some object type implementations stringify as their bare name, others do not.
        for name in ("rover", "lander", "objective", "sample", "obstacle"):
            try:
                self.types[name] = observation_space.get_type(name)
            except Exception:
                pass

    def _objects(self, state, name):
        return list(state.get_objects(self.types[name]))

    @staticmethod
    def _v(state, obj, feature):
        return float(state.get(obj, feature))

    def _xy(self, state, obj):
        return (self._v(state, obj, "x"), self._v(state, obj, "y"))

    def reset(self, state, info):
        rovers = sorted(self._objects(state, "rover"), key=lambda x: x.name)
        self.home = [self._xy(state, r) for r in rovers]
        self.assignments = [[], []]
        self.sample_targets = [[], []]
        self.sample_stands = {}
        self.mode = ["work", "work"]
        self.radio_attempt = [0, 0]
        self.image_phase = [0, 0]
        self.stalls = [0, 0]
        self.last_xy = list(self.home)
        self.detours = [None, None]
        self.boxes = []
        for o in self._objects(state, "obstacle"):
            self.boxes.append((self._v(state, o, "x"), self._v(state, o, "y"),
                               self._v(state, o, "half_x"), self._v(state, o, "half_y"),
                               self._v(state, o, "half_z")))
        lander = self._objects(state, "lander")[0]
        self.lander_xy = self._xy(state, lander)
        self._build_assignments(state)

    def _build_assignments(self, state):
        objectives = self._objects(state, "objective")
        # Keep each rover on its side when possible. Central mounds go to the nearer home.
        for o in objectives:
            x, y = self._xy(state, o)
            costs = [math.hypot(x-h[0], y-h[1]) + (7.0 if x*h[0] < -0.05 else 0.0)
                     for h in self.home]
            self.assignments[0 if costs[0] <= costs[1] else 1].append(o.name)

        samples = self._objects(state, "sample")
        sample_points = {s.name: self._xy(state, s) for s in samples}
        by_kind = {0: [], 1: []}
        for s in samples:
            by_kind[int(self._v(state, s, "is_soil") > 0.5)].append(s)
        # Choose each required kind independently. A rover may collect both kinds;
        # this is necessary on layouts where one side contains all usable samples.
        for kind in (0, 1):
            choices = []
            for s in by_kind[kind]:
                p = self._xy(state, s)
                for i in range(2):
                    # Samples close to the wall are reachable from the matching side
                    # without putting the rover center across it.
                    matching_side = (p[0] >= 0.0) == (self.home[i][0] >= 0.0)
                    if matching_side:
                        stands = []
                        if not self._blocked(state, p, i):
                            stands.append(p)
                        for radius in (.18, .235):
                            for k in range(24):
                                a = 2*math.pi*k/24
                                q = (p[0]+radius*math.cos(a), p[1]+radius*math.sin(a))
                                if not self._blocked(state, q, i):
                                    stands.append(q)
                        # Sampling selects the nearest sample. Avoid a pose where a
                        # nearby sample of the other kind would be selected instead.
                        stands = [q for q in stands if all(
                            other == s.name or self._dist(q, p) + .005 < self._dist(q, op)
                            for other, op in sample_points.items())]
                        if stands:
                            stand = min(stands, key=lambda q: self._dist(self.home[i], q))
                            self.sample_stands[(i, s.name)] = stand
                            choices.append((self._dist(self.home[i], stand), i, s.name))
            if choices:
                _, i, name = min(choices)
                self.sample_targets[i].append(name)
        for i in range(2):
            # Objectives are on the northern mound row; visit northern samples
            # first while returning south from that row.
            self.sample_targets[i].sort(
                key=lambda n: -self.sample_stands[(i, n)][1])

        # Greedy nearest-neighbour order is adequate for the small objective sets.
        for i in range(2):
            cur = self.home[i]
            names = self.assignments[i]
            ordered = []
            while names:
                name = min(names, key=lambda n: self._dist(cur, self._xy(state, state.get_object_from_name(n))))
                names.remove(name); ordered.append(name)
                cur = self._xy(state, state.get_object_from_name(name))
            self.assignments[i] = ordered

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _blocked(self, state, p, rover_index):
        x, y = p
        if abs(x) > 2.26 or abs(y) > 2.26:
            return True
        # Fixed wall; the rover can radio around its short southern endpoint but
        # should otherwise remain in its own chamber.
        if abs(x) < 0.27 and -2.30 < y < 1.62:
            return True
        for ox, oy, bx, by, bz in self.boxes:
            if bz > .15:  # fixed headings give an anisotropic rover footprint
                if abs(x-ox) < .255 and abs(y-oy) < .29:
                    return True
            else:
                hx = bx + 0.24
                hy = by + 0.24
                if abs(x-ox) < hx and abs(y-oy) < hy:
                    return True
        lx, ly = self.lander_xy
        # Husky chassis is substantially larger than the thin pillars.
        if abs(x-lx) < 0.75 and abs(y-ly) < 0.60:
            return True
        return False

    @staticmethod
    def _segment_box(a, b, cx, cy, hx, hy):
        # Liang-Barsky intersection; endpoints touching count as blocked.
        dx, dy = b[0]-a[0], b[1]-a[1]
        lo, hi = 0.0, 1.0
        for q, d, mn, mx in ((a[0], dx, cx-hx, cx+hx), (a[1], dy, cy-hy, cy+hy)):
            if abs(d) < 1e-9:
                if q < mn or q > mx:
                    continue
                continue
            t0, t1 = (mn-q)/d, (mx-q)/d
            if t0 > t1:
                t0, t1 = t1, t0
            lo, hi = max(lo, t0), min(hi, t1)
            if lo > hi:
                return False
        return True

    def _los_guess(self, state, p, target, target_name=None):
        if self._dist(p, target) > 1.94:
            return False
        # The wall is tall; mounds are low enough to see their own objective.
        if self._segment_box(p, target, 0.0, -0.25, 0.02, 1.75):
            return False
        for ox, oy, hx, hy, hz in self.boxes:
            # Ignore the low mound supporting this target, but not tall pillars.
            if hz < 0.15 and abs(target[0]-ox) <= hx+.05 and abs(target[1]-oy) <= hy+.05:
                continue
            if hz > 0.15 and self._segment_box(p, target, ox, oy, hx+.02, hy+.02):
                return False
        return True

    def _vantage(self, state, rover_index, target_obj):
        target = self._xy(state, target_obj)
        current = self._xy(state, self._objects(state, "rover")[rover_index])
        candidates = []
        radii = (.9, 1.1) if self.stalls[rover_index] > 3 else (1.45, 1.75, 1.1)
        for radius in radii:
            for k in range(24):
                ang = 2*math.pi*k/24
                p = (target[0]+radius*math.cos(ang), target[1]+radius*math.sin(ang))
                if not self._blocked(state, p, rover_index) and self._los_guess(state, p, target, target_obj.name):
                    candidates.append(p)
        if not candidates:
            return (target[0], target[1]-1.4)
        # Prefer current-side, short travel positions.
        return min(candidates, key=lambda p: self._dist(current, p) + (3.0 if p[0]*self.home[rover_index][0] < -.05 else 0.0))

    def _next_motion(self, state, rover_index, goal, tolerance):
        rover = sorted(self._objects(state, "rover"), key=lambda x: x.name)[rover_index]
        start = self._xy(state, rover)
        if self._dist(start, goal) <= tolerance:
            return (0.0, 0.0, True)
        # Finish at the exact continuous waypoint and apply the operator on that
        # same step (the environment checks operators after translation).
        if abs(goal[0]-start[0]) <= .2 and abs(goal[1]-start[1]) <= .2 and not self._blocked(state, goal, rover_index):
            return (goal[0]-start[0], goal[1]-start[1], True)
        # If the simulator rejected a modeled-safe step, cycle through nearby
        # alternatives. This handles the rover's rounded, anisotropic footprint.
        if self.stalls[rover_index] >= 3:
            alts = []
            for dx, dy in ((-.2,-.2),(-.2,0),(-.2,.2),(0,-.2),(0,.2),(.2,-.2),(.2,0),(.2,.2)):
                q = (start[0]+dx, start[1]+dy)
                if not self._blocked(state, q, rover_index):
                    alts.append((self._dist(q, goal), dx, dy))
            alts.sort()
            if alts:
                _, dx, dy = alts[(self.stalls[rover_index]-3) % len(alts)]
                return (dx, dy, False)

        # A* on a 0.2 m lattice anchored at the current pose. Nodes may move on both axes.
        step = 0.2
        sx, sy = start
        def pos(node): return (sx + step*node[0], sy + step*node[1])
        def heuristic(node):
            p = pos(node)
            return max(abs(p[0]-goal[0]), abs(p[1]-goal[1])) / step
        openq = [(heuristic((0, 0)), 0, (0, 0))]
        came = {}; bestg = {(0, 0): 0}; end = None
        counter = 0
        while openq and counter < 3500:
            _, g, node = heapq.heappop(openq); counter += 1
            if g != bestg.get(node):
                continue
            p = pos(node)
            if self._dist(p, goal) <= max(tolerance, .18):
                end = node; break
            for dx, dy in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                nn = (node[0]+dx, node[1]+dy); np_ = pos(nn)
                if self._blocked(state, np_, rover_index):
                    continue
                ng = g+1
                if ng < bestg.get(nn, 10**9):
                    bestg[nn] = ng; came[nn] = node
                    heapq.heappush(openq, (ng+heuristic(nn), ng, nn))
        if end is None:
            # Direct fallback also lets feedback expose an overly conservative model.
            dx = max(-.2, min(.2, goal[0]-sx)); dy = max(-.2, min(.2, goal[1]-sy))
            return (dx, dy, False)
        if end == (0, 0):
            return (0.0, 0.0, True)
        while came.get(end) != (0, 0) and end in came:
            end = came[end]
        p = pos(end)
        return (p[0]-sx, p[1]-sy, False)

    def _all_received(self, state):
        images = all(self._v(state, o, "received_image") > .5 for o in self._objects(state, "objective"))
        kinds = [False, False]
        for s in self._objects(state, "sample"):
            if self._v(state, s, "received_analysis") > .5:
                kinds[int(self._v(state, s, "is_soil") > .5)] = True
        return images and all(kinds)

    def _has_unreceived(self, state, i):
        have = "have_image_rover0" if i == 0 else "have_image_rover1"
        analyzed = "analyzed_rover0" if i == 0 else "analyzed_rover1"
        for o in self._objects(state, "objective"):
            if self._v(state, o, have) > .5 and self._v(state, o, "received_image") < .5:
                return True
        for s in self._objects(state, "sample"):
            if self._v(state, s, analyzed) > .5 and self._v(state, s, "received_analysis") < .5:
                return True
        return False

    def _rover_command(self, state, i):
        rovers = sorted(self._objects(state, "rover"), key=lambda x: x.name)
        rover = rovers[i]
        here = self._xy(state, rover)
        moved = self._dist(here, self.last_xy[i]) > .03
        self.stalls[i] = 0 if moved else self.stalls[i]+1
        self.last_xy[i] = here

        # Remove objectives already photographed by either rover.
        self.assignments[i] = [n for n in self.assignments[i]
                               if self._v(state, state.get_object_from_name(n), "received_image") < .5
                               and self._v(state, state.get_object_from_name(n), "have_image_rover0") < .5
                               and self._v(state, state.get_object_from_name(n), "have_image_rover1") < .5]

        # All objectives lie on the northern mound row. Imaging them first turns
        # the remaining sample route into a mostly one-way trip back toward home.
        if self.assignments[i]:
            obj = state.get_object_from_name(self.assignments[i][0])
            goal = self._vantage(state, i, obj)
            dx, dy, arrived = self._next_motion(state, i, goal, .12)
            if not arrived:
                self.image_phase[i] = 0
                return dx, dy, NOOP
            calibrated = self._v(state, rover, "calibrated") > .5
            return 0.0, 0.0, IMAGE if calibrated else CALIBRATE

        analyzed_feature = "analyzed_rover0" if i == 0 else "analyzed_rover1"
        self.sample_targets[i] = [n for n in self.sample_targets[i]
                                  if self._v(state, state.get_object_from_name(n), analyzed_feature) < .5]
        if self.sample_targets[i]:
            # Stores hold one physical sample. Dropping does not erase its analysis.
            if self._v(state, rover, "store_full") > .5:
                return 0.0, 0.0, DROP
            name = self.sample_targets[i][0]
            goal = self.sample_stands[(i, name)]
            dx, dy, arrived = self._next_motion(state, i, goal, .005)
            return dx, dy, SAMPLE if arrived else NOOP

        # The wall blocks rover0's (right-side) home from the lander. A small move
        # south sees around the lower endpoint, avoiding a very long crossing tour.
        if i == 0 and self._has_unreceived(state, i):
            priority = [(x, y) for y in (-1.8, -1.9, -2.16, -2.20, -1.2, -.5, .2)
                        for x in (.8, .55, 1.0, 1.3, 1.6, 1.9)]
            candidates = [p for p in priority if not self._blocked(state, p, i)]
            radio_point = candidates[(self.radio_attempt[i]//2) % len(candidates)] if candidates else (1.0, -2.2)
            dx, dy, arrived = self._next_motion(state, i, radio_point, .15)
            if not arrived:
                return dx, dy, NOOP
            self.radio_attempt[i] += 1
            return dx, dy, SEND

        # Everything assigned to this rover has been acquired. Return, send, drop.
        # When close, command the exact known-safe home pose. This avoids lattice
        # quantization just outside the strict 0.25 m home predicate.
        if abs(self.home[i][0]-here[0]) <= .2 and abs(self.home[i][1]-here[1]) <= .2:
            dx, dy, arrived = self.home[i][0]-here[0], self.home[i][1]-here[1], True
        else:
            dx, dy, arrived = self._next_motion(state, i, self.home[i], .09)
        if not arrived:
            return dx, dy, NOOP
        if self._has_unreceived(state, i):
            return dx, dy, SEND
        if self._v(state, rover, "store_full") > .5:
            return dx, dy, DROP
        return 0.0, 0.0, SEND

    def get_action(self, state):
        action = np.zeros(8, dtype=np.float32)
        for i in range(2):
            dx, dy, op = self._rover_command(state, i)
            base = 4*i
            action[base] = max(-.2, min(.2, dx))
            action[base+1] = max(-.2, min(.2, dy))
            action[base+2] = 0.0
            action[base+3] = op
        return action

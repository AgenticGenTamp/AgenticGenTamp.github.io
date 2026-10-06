import json, math
import numpy as np
from layout import *
S = ("cuboid", (0.05, 0.05)); E = ("tri", ("equilateral", 0.1, 0.1)); R = ("tri", ("right", 0.1, 0.1))
D = {"S": S, "E": E, "R": R}
xmin, ymin, xmax, ymax = RACK_INNER; cx, cy = 0.3, 0.0; h = math.sqrt(3) / 2 * 0.1; t = 0.1 / 3
sq = [(cx + 0.04, cy - 0.09, 0.0), (cx - 0.04, cy + 0.09, 0.0)]
eq = [(xmax - h / 3, cy + 0.01, math.pi / 2), (xmin + h / 3, cy - 0.01, -math.pi / 2)]
rt = [(xmax - t, cy - 0.04 + t, math.pi / 2), (xmin + t, cy + 0.04 - t, -math.pi / 2)]
out = {}
for key in ["R", "RS", "ERS", "RRS", "ERSS", "RRSS"]:
    parts = [D[c] for c in key]; fps = [local_footprint(k, p) for k, p in parts]
    init = []; si = ti = 0
    for c in key:
        if c == "S":
            init.append(sq[si]); si += 1
        else:
            init.append((eq if c == "E" else rt)[ti]); ti += 1
    best = None
    for seed in range(4):
        lay, c, P = refine_layout(fps, RACK_INNER, init, iters=3000, rng=np.random.default_rng(seed))
        lay2, c2, _ = refine_layout(fps, RACK_INNER, P, iters=4000, rng=np.random.default_rng(50 + seed),
                                    step0=(0.003, 0.003, 0.03))
        if c2 > c:
            lay, c = lay2, c2
        if best is None or c > best[1]:
            best = (lay, c)
    print(key, round(best[1], 4), [round(o[2], 2) for o in best[0]], flush=True)
    out[key] = [list(map(float, o)) for o in best[0]]
json.dump(out, open("layouts_mirror.json", "w"), indent=1)

import numpy as np, math
from expD_lib import *
from expD_core import approach_grasp

def table_clear(bx, by):
    dx = max(4.2-bx, bx-4.8, 0.0); dy = max(-0.6-by, by-0.6, 0.0)
    return math.hypot(dx, dy)

def candidates(blk, d, backs=(0.90,0.85,0.95), lats=(-0.188,0.0,0.188,-0.30,0.30),
               dyaws=(0.0,0.3,-0.3), clear=0.36):
    out = []
    for bk in backs:
        for la in lats:
            for dy in dyaws:
                b = base_for(blk, d, bk, la, dy)
                if max(abs(b[0]), abs(b[1])) > 4.98: continue
                c = table_clear(b[0], b[1])
                if c < clear: continue
                out.append((bk, la, dy, b, c))
    return out

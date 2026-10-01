"""Far-hook grasp recipe (hook_x > 3.40, out of reach of the standard top-of-post grasp).

Verified with far_e7.py: HELD=1 on seeds 1,8,9,14,35,41,43,44,45,50 (hook_x 3.416..3.490).
Fails on seeds 53 (3.4749) and 60 (3.4964): the jaw simply cannot reach that far.

Key geometry (measured):
  robot x max          = 3.2965 (base r=0.20, wall inner face ~3.4965)
  max arm tilt at x_max, arm=0.4, gap=0.25 : phi = 0.098 rad (wall blocks further rotation)
  jaw centre           = (3.2965 + 0.46*sin(phi), y - 0.46*cos(phi))  -> jaw_x = 3.345
  right finger inner face = jaw_x + 0.125*cos(phi) = 3.4696  == max graspable hook_x
"""
import math
PI = math.pi
PHI = 0.098          # max wall-permitted tilt with arm=0.4, gap=0.25
ARM = 0.4
JAW_Y = 0.46         # jaw centre 0.04 below the post top (post top = hook_y+0.5)

def far_grasp(s):
    """s: scoop_lib.Scoop. Returns True if hook 'held'==1. ~80-145 steps."""
    s.goto(y=2.3, th=-PI/2, arm=ARM, gap=0.25, tol=0.01, maxit=70)   # travel pose, above divider
    s.goto(x=3.30, y=2.3, tol=0.01, maxit=90)                        # clamps at x=3.2965
    s.goto(y=1.2, tol=0.01, maxit=50)
    s.goto(th=-PI/2 + PHI, tol=0.01, maxit=15)                       # tilt jaw toward the wall
    th = s.g('theta') + PI/2                                         # achieved tilt
    hy = s.g('y', s.H)
    s.goto(y=hy + JAW_Y + (ARM+0.06)*math.cos(th), tol=0.01, maxit=40)
    for _ in range(14):                                              # close on the post
        s.step([0, 0, 0, 0, -0.015])
        if s.g('held', s.H) > 0.5:
            return True
    return s.g('held', s.H) > 0.5

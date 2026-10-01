import numpy as np, ctrl
from env_client import make_env

class E:
    def __init__(self, seed=0):
        self.env = make_env(); self.obs, _ = self.env.reset(seed=seed)
    def r(self):
        return ctrl.rob(self.obs)
    def step(self, a):
        prev = self.r()
        self.obs, _, _, _, _ = self.env.step(np.asarray(a, dtype=np.float32))
        cur = self.r()
        rej = np.allclose(prev[:10], cur[:10]) and np.abs(np.asarray(a)[:10]).max() > 1e-9
        return rej
    def move_base(self, dx=0.0, dy=0.0, dr=0.0):
        a = np.zeros(11); a[0] = dx; a[1] = dy; a[2] = dr
        return self.step(a)
    def goto_y(self, y, stepsz=0.1):
        # move in y only, at current x
        while True:
            cy = self.r()[1]
            d = y - cy
            if abs(d) < 1e-6: return True
            rej = self.move_base(dy=np.clip(d, -stepsz, stepsz))
            if rej: return False
    def goto_x(self, x, stepsz=0.1):
        while True:
            cx = self.r()[0]
            d = x - cx
            if abs(d) < 1e-6: return True
            rej = self.move_base(dx=np.clip(d, -stepsz, stepsz))
            if rej: return False
    def push_x(self, stepsz=0.05, limit=1.5, sign=+1):
        n = 0
        while n < 200:
            cx = self.r()[0]
            if sign > 0 and cx > limit: break
            if sign < 0 and cx < -limit: break
            rej = self.move_base(dx=sign*stepsz)
            n += 1
            if rej: return self.r()[0], True
        return self.r()[0], False
    def close(self):
        self.env.close()

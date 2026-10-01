import numpy as np
from env_client import make_env

class E:
    def __init__(self, seed=0):
        self.env=make_env(); self.o,_=self.env.reset(seed=seed); self.n=0
    def move(self, dx, dy):
        a=np.zeros(11); a[0]=dx; a[1]=dy
        o2,r,t,tr,_=self.env.step(a); self.n+=1
        moved = np.linalg.norm(o2[:2]-self.o[:2])>1e-9
        self.o=o2
        return moved, t
    def goto(self, gx, gy, step=0.1):
        for k in range(500):
            dx=float(np.clip(gx-self.o[0],-step,step)); dy=float(np.clip(gy-self.o[1],-step,step))
            if abs(dx)<1e-7 and abs(dy)<1e-7: return True
            m,t=self.move(dx,dy)
            if not m: return False
            if t: return True
        return False
    def pos(self): return float(self.o[0]), float(self.o[1])

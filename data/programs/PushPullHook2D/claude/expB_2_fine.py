import numpy as np
from helper import Sim
def probe(s,d,steps=(0.005,0.001,0.0002,0.00005),maxn=800):
    d=np.array(d,float)
    for st in steps:
        for i in range(maxn):
            p=s.obs[:2].copy(); s.step([d[0]*st,d[1]*st,0,0,0.0])
            if np.linalg.norm(s.obs[:2]-p)<1e-12: break
    return s.obs[:2].copy()

# 1) top limit at several x, arm/theta variations
print("--- +y limit vs x, arm, theta")
for x in [0.9,2.2,3.0]:
    for arm in [0.1,0.2]:
        for th in [np.pi/2,-np.pi/2]:
            s=Sim(42); s.goto(x,0.7,th,arm,0.0)
            p=probe(s,(0,1)); print(f"x={x} arm={arm} th={th:+.2f} ymax={p[1]:.5f}"); s.close()
print("--- -y limit")
for arm in [0.1,0.2]:
    for th in [np.pi/2,-np.pi/2]:
        s=Sim(42); s.goto(2.2,0.7,th,arm,0.0); p=probe(s,(0,-1))
        print(f"arm={arm} th={th:+.2f} ymin={p[1]:.5f}"); s.close()
print("--- +x limit vs arm (theta=0 facing +x)")
for arm in [0.1,0.12,0.15,0.18,0.2]:
    s=Sim(42); s.goto(2.2,0.7,0.0,arm,0.0); p=probe(s,(1,0))
    print(f"arm={arm} th=0 xmax={p[0]:.5f} armobs={s.obs[4]:.4f}"); s.close()
s=Sim(42); s.goto(2.2,0.7,np.pi,0.2,0.0); p=probe(s,(1,0)); print("arm=0.2 th=pi xmax=%.5f"%p[0]); s.close()
print("--- diagonal theta effect: theta=45deg, move +x and +y, arm=0.2")
for th in [np.pi/4, np.pi/8, 3*np.pi/8]:
    s=Sim(42); s.goto(2.2,0.7,th,0.2,0.0); p=probe(s,(1,0)); print(f"th={th:.3f} xmax={p[0]:.5f}"); s.close()
    s=Sim(42); s.goto(2.2,0.7,th,0.2,0.0); p=probe(s,(0,1)); print(f"th={th:.3f} ymax={p[1]:.5f}"); s.close()

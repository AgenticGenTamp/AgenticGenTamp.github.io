import numpy as np
from helper import E
res=[]
for x in np.arange(-2.2,2.25,0.1):
    e=E(0)
    ok=e.goto(float(x), -1.5, step=0.2)
    y=-1.5
    while True:
        m,t=e.move(0,-0.01)
        if not m: break
        y-=0.01
        if y< -4: break
    res.append((round(float(x),2), round(y,3)))
print(res)
T=np.load("targets.npy")
print("target x range", T[:,0].min(), T[:,0].max(), "y", T[:,1].min(), T[:,1].max())

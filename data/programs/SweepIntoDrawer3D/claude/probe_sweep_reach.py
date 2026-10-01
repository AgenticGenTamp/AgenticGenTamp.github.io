import numpy as np
from ctrl2 import tip_to_fk, RDOWN, JLO, JHI
from ik import ik
def w2l(base,wx,wy):
    bx,by,yaw=base; dx,dy=wx-bx,wy-by
    return np.cos(yaw)*dx+np.sin(yaw)*dy, -np.sin(yaw)*dx+np.cos(yaw)*dy
q0=np.array([0,-0.35,3.14,-2.55,0,-0.87,1.57])
for bx in [1.242,1.6]:
    base=[bx,-0.084,3.096]
    print(f"=== base x={bx}")
    for z in [0.466,0.60]:
        rows=[]
        for x in np.arange(0.40,1.25,0.02):
            ys=[]
            for y in np.arange(-0.60,0.35,0.02):
                lx,ly=w2l(base,x,y)
                p=tip_to_fk([lx,ly,z],RDOWN)
                q,res=ik(np.array(p),RDOWN,q0)
                if res<1e-3 and np.abs(np.clip(q,JLO,JHI)-q).max()<1e-6: ys.append(y)
            if ys: rows.append((round(float(x),2),round(min(ys),2),round(max(ys),2)))
        xs=[r[0] for r in rows]
        print(f" z={z}: x in [{min(xs)},{max(xs)}]")
        print("   ", " ".join(f"{r[0]}:[{r[1]},{r[2]}]" for r in rows[::2]))

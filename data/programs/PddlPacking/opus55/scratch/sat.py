import numpy as np
def sat(c1,y1,c2,y2,h=0.035):
    d=np.array(c2)-np.array(c1); axes=[]
    for y in (y1,y2): axes+= [np.array([np.cos(y),np.sin(y)]),np.array([-np.sin(y),np.cos(y)])]
    gap=[]
    for a in axes:
        r1=h*(abs(np.cos(y1)*a[0]+np.sin(y1)*a[1])+abs(-np.sin(y1)*a[0]+np.cos(y1)*a[1]))
        r2=h*(abs(np.cos(y2)*a[0]+np.sin(y2)*a[1])+abs(-np.sin(y2)*a[0]+np.cos(y2)*a[1]))
        gap.append(abs(d@a)-(r1+r2))
    return max(gap)
def aabb(c1,y1,c2,y2,h=0.035):
    e=lambda y:h*(abs(np.cos(y))+abs(np.sin(y)))
    d=np.abs(np.array(c2)-np.array(c1)); return max(d-(e(y1)+e(y2)))
A=(-0.05,0)
for c,y,res in [((0.03,0),0.401,"refused"),((0.034,0),0.413,"accepted"),((0.0101,0.0601),0.472,"refused"),((0.0136,0.0636),0.492,"accepted")]:
    print(res,"OBB gap",round(sat(A,0,c,y),4),"AABB gap",round(aabb(A,0,c,y),4))

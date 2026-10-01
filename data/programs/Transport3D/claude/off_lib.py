import numpy as np, kutil, kin
kutil.MZ=0.269
def fkpos(c):
    b,q,g=c.robot(); ax,ay=c.armbase(b)
    T=kin.fk(q,base_x=ax,base_y=ay,base_rot=b[2],mount=(0,0,0.269))
    return T,q,b
def cmove(c,tgt,yaw,step=0.04):
    """cartesian straight-line move in small steps from current fk pos"""
    T,q,b=fkpos(c); cur=T[:3,3].copy(); tgt=np.asarray(tgt,float)
    n=max(1,int(np.ceil(np.linalg.norm(tgt-cur)/step)))
    for i in range(1,n+1):
        p=cur+(tgt-cur)*i/n
        if not c.move_to(p,yaw=yaw): return False
    return True
def safe_to(c,x,y,z,yaw,hi=0.28):
    T,_,_=fkpos(c)
    if T[2,3]<hi-1e-3:
        if not cmove(c,[T[0,3],T[1,3],hi],yaw): return False
    if not cmove(c,[x,y,hi],yaw): return False
    return cmove(c,[x,y,z],yaw)

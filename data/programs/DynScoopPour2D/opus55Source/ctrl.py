import numpy as np
LIM=np.array([0.03,0.03,np.pi/32,0.08,0.015])*0.9999
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def toward(r, tgt):
    """r: dict robot; tgt: (x,y,theta,arm,gap) entries may be None"""
    cur=[r['x'],r['y'],r['theta'],r['arm_joint'],r['finger_gap']]
    a=np.zeros(5)
    for i,t in enumerate(tgt):
        if t is None: continue
        d = wrap(t-cur[i]) if i==2 else t-cur[i]
        a[i]=d
    # scale xy jointly
    n=np.hypot(a[0],a[1])
    if n>0.03: a[:2]*=0.03/n
    a[2:]=np.clip(a[2:],-LIM[2:],LIM[2:])
    return np.clip(a,-LIM,LIM).astype(np.float32)
def reached(r,tgt,tol=1e-3):
    cur=[r['x'],r['y'],r['theta'],r['arm_joint'],r['finger_gap']]
    for i,t in enumerate(tgt):
        if t is None: continue
        d = wrap(t-cur[i]) if i==2 else t-cur[i]
        if abs(d)>tol: return False
    return True

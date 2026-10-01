import numpy as np, fk, ctrl
q0=np.array([0.6772,-0.3431,1.2,-1.4669,1.2422,-1.9544,2.2225])
SEEDS=[q0, np.array([0.3,-0.4,1.3,-0.7,1.5,-2.0,3.3]), np.array([1.5,0.5,2.0,-1.5,0.0,-1.0,0.0]),
       np.array([0.0,0.0,1.5,-1.0,0.0,-1.5,0.0]), np.array([-0.5,0.3,0.5,-2.0,1.0,-1.0,1.0])]
def targR_tilt(ang, tilt, tiltdir):
    """approach axis (tool x) tilted by `tilt` from -z, tilted toward direction tiltdir (world angle)"""
    x=np.array([np.sin(tilt)*np.cos(tiltdir), np.sin(tilt)*np.sin(tiltdir), -np.cos(tilt)])
    y=np.array([np.cos(ang),np.sin(ang),0.0])
    y=y-x*np.dot(x,y); y/=np.linalg.norm(y)
    z=np.cross(x,y)
    return np.column_stack([x,y,z])
def reachable(pw, base, ang, tilt=0.0, tiltdir=0.0):
    bx,by,rot=base
    c,s=np.cos(rot),np.sin(rot); Rb=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    tp=Rb.T@(pw-np.array([bx,by,0]))
    Rw=targR_tilt(ang,tilt,tiltdir); Rl=Rb.T@Rw
    for sd in SEEDS:
        q,ok=fk.ik(tp,Rl,sd,iters=150)
        if ok: return q
    return None
BASES=[(-0.43,by,0.0) for by in np.arange(-0.22,0.23,0.11)]+[(-0.64,by,0.0) for by in np.arange(-0.9,0.91,0.1)]
grid=[]
for byy in np.arange(-0.53,0.54,0.0662):
    row=""
    for bxx in np.arange(-0.265,0.266,0.0663):
        pw=np.array([bxx,byy,0.831])
        found=False
        for base in BASES:
            for ang in [np.pi/2,0.0]:
                if reachable(pw,base,ang) is not None: found=True;break
            if found:break
        row+="#" if found else "."
    grid.append((byy,row))
for byy,row in grid: print(f"{byy:+.2f} {row}")
print("x from -0.265 to 0.265")

print("=== corner analysis ===")
for pw in [np.array([0.265,-0.53,0.831]),np.array([0.2,-0.5,0.831]),np.array([0.265,-0.4,0.831])]:
    opts=[]
    for rot in [0.0,-0.2,-0.4,-0.6,0.2]:
        for by in np.arange(-0.22,0.23,0.11):
            for tilt in [0.0,0.35,0.6]:
                for tiltdir in ([0.0] if tilt==0 else [np.arctan2(-0.53-by,0.265+0.43)+np.pi, np.pi, -np.pi/2]):
                    for ang in [np.pi/2,0.0,np.pi/4]:
                        if reachable(pw,(-0.43,by,rot),ang,tilt,tiltdir) is not None:
                            opts.append((rot,round(by,2),round(tilt,2),round(tiltdir,2),round(ang,2)))
    print(np.round(pw,2), len(opts), opts[:6])

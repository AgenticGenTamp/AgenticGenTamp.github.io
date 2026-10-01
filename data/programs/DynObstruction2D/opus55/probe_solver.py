import numpy as np
W=3.236; FLOOR=0.1
def rects(x,y,th,L,g):
    d=np.array([np.cos(th),np.sin(th)]); p=np.array([-np.sin(th),np.cos(th)])
    B=np.array([x,y]); A=B+L*d
    out=[]
    for c,hd,hp in [(A,0.03,0.16),(A+0.1*d+g/2*p,0.1,0.03),(A+0.1*d-g/2*p,0.1,0.03)]:
        out.append(np.array([c+sd*hd*d+sp*hp*p for sd,sp in [(1,1),(1,-1),(-1,-1),(-1,1)]]))
    return out
def rect_hits_box(poly,x0,x1,y0,y1,eps=1e-3):
    # SAT between convex quad and AABB
    axes=[np.array([1,0]),np.array([0,1])]
    for i in range(2):
        e=poly[(i+1)%4]-poly[i]; axes.append(np.array([-e[1],e[0]])/np.linalg.norm(e))
    box=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    for ax in axes:
        a=poly@ax; b=box@ax
        if a.max()<=b.min()+eps or b.max()<=a.min()+eps: return False
    return True
def collides(pose, blk, eps=1e-3):
    x,y,th,L,g=pose
    bx0,bx1,by1=blk  # block box x0,x1, top y
    if x<0.24+eps or x>W-0.24-eps: return True
    # base circle vs block
    cx=min(max(x,bx0),bx1); cy=min(max(y,FLOOR),by1)
    if np.hypot(x-cx,y-cy)<0.24+eps: return True
    if y<FLOOR+0.24: return True
    for r in rects(*pose):
        if r[:,0].min()<eps or r[:,0].max()>W-eps or r[:,1].min()<FLOOR+eps: return True
        if rect_hits_box(r,bx0,bx1,FLOOR,by1,eps): return True
    return False
def best_pose(blk, side='L', ths=None, gs=(0.12,0.2,0.32), Ls=(0.24,0.36,0.48)):
    """block near left wall: slot (0,bx0). returns list of (D, pose) sorted desc"""
    bx0,bx1,yt=blk
    res=[]
    if ths is None: ths=np.linspace(-np.pi,0,73)
    for th in ths:
      for g in gs:
        for L in Ls:
          for x in np.arange(0.24,bx1+0.8,0.01):
            # find lowest y from above, descending
            y=yt+0.24+L+0.25
            if collides((x,y,th,L,g),blk): continue
            lo=None
            for yy in np.arange(y,0.3,-0.005):
                if collides((x,yy,th,L,g),blk): break
                lo=yy
            if lo is None: continue
            pose=(x,lo,th,L,g)
            V=np.vstack(rects(*pose)); iy=V[:,1].argmin()
            if V[iy,0]>=bx0 or V[iy,1]>yt: continue
            # part in slot below top: max x of vertices below top must be < bx0
            D=yt-V[iy,1]
            res.append((D,pose))
    res.sort(key=lambda t:-t[0])
    return res
if __name__=='__main__':
    import sys
    for e in [0.08,0.12,0.16,0.2,0.25,0.3,0.35,0.4,0.49]:
        blk=(e,e+0.4,0.1+0.3)
        r=best_pose(blk,ths=np.linspace(-np.pi,0,37),gs=(0.12,0.32),Ls=(0.24,0.48))
        print(e, [ (round(D,3),tuple(np.round(p,3))) for D,p in r[:2]])

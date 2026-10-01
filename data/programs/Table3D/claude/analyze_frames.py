import subprocess, numpy as np, glob, os
W,H=640,360
def load(p):
    raw=subprocess.run(['convert',p,'-depth','8','rgb:-'],capture_output=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(H,W,3).astype(int)
files=sorted(glob.glob('mcp_renders/state_custom_policy_seed0_000?_1.png'))
def comps(mask,minsize=8):
    # simple flood fill labeling
    lab=-np.ones(mask.shape,int); out=[]; cid=0
    ys,xs=np.nonzero(mask)
    from collections import deque
    for y0,x0 in zip(ys,xs):
        if lab[y0,x0]!=-1: continue
        q=deque([(y0,x0)]); lab[y0,x0]=cid; pts=[]
        while q:
            y,x=q.popleft(); pts.append((y,x))
            for dy in(-1,0,1):
                for dx in(-1,0,1):
                    ny,nx=y+dy,x+dx
                    if 0<=ny<mask.shape[0] and 0<=nx<mask.shape[1] and mask[ny,nx] and lab[ny,nx]==-1:
                        lab[ny,nx]=cid; q.append((ny,nx))
        if len(pts)>=minsize:
            a=np.array(pts); out.append(dict(n=len(pts),y0=a[:,0].min(),y1=a[:,0].max(),x0=a[:,1].min(),x1=a[:,1].max(),cy=a[:,0].mean(),cx=a[:,1].mean()))
        cid+=1
    return sorted(out,key=lambda d:d['cx'])
for f in files:
    im=load(f); R,G,B=im[...,0],im[...,1],im[...,2]
    purple=(R>35)&(G<45)&(B>35)&(np.abs(R-B)<25)&(R>2*G+10)
    # light gray fingertip: bright & neutral
    tip=(R>180)&(np.abs(R-G)<12)&(np.abs(G-B)<12)&(np.abs(R-B)<12)
    print('==',os.path.basename(f))
    print('  cubes:',[(c['n'],round(c['cx'],1),round(c['cy'],1),c['x0'],c['x1'],c['y0'],c['y1']) for c in comps(purple,20)])
    print('  tips :',[(c['n'],round(c['cx'],1),round(c['cy'],1),c['x0'],c['x1'],c['y0'],c['y1']) for c in comps(tip,15)])

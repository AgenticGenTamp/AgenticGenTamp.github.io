import numpy as np
import heapq

def path_translation(s,goal,clearance=.12):
    """Fast planar path for a hook at a fixed orientation."""
    u=np.array([np.cos(s[11]),np.sin(s[11])]);v=np.array([-u[1],u[0]])
    offset=s[9:11]-s[:2]
    ends=np.array([offset,offset-s[18]*u,offset-s[19]*v])
    low=np.maximum([.105,.105],.045-ends.min(axis=0))
    high=np.minimum([3.395,1.145],[3.455,2.455]-ends.max(axis=0))
    if np.any(low>=high):return None
    axes=[np.linspace(low[j],high[j],max(2,int((high[j]-low[j])/.03)+1)) for j in range(2)]
    nx,ny=len(axes[0]),len(axes[1]);xx,yy=np.meshgrid(*axes,indexing='ij')
    pts=np.stack([xx.ravel(),yy.ravel()],axis=1)
    def free(p):
        p=np.asarray(p).reshape(-1,2)
        good=np.all((p>=low-1e-5)&(p<=high+1e-5),axis=1)
        b=s[20:22]-p-offset
        for vec in [-s[18]*u,-s[19]*v]:
            t=np.clip((b@vec)/(vec@vec),0,1)
            good &= np.linalg.norm(b-t[:,None]*vec,axis=1)>clearance
        return good
    valid=free(pts)
    start=np.argmin(np.linalg.norm(pts-s[:2],axis=1));end=np.argmin(np.linalg.norm(pts-goal,axis=1))
    valid[start]=True
    if not valid[end]:return None
    q=[(float(np.linalg.norm(pts[start]-goal)),0.,int(start))];cost={int(start):0.};prev={}
    found=False
    while q:
        _,g,n=heapq.heappop(q)
        if n==end:found=True;break
        if g>cost[n]+1e-9:continue
        x,y=divmod(n,ny)
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ix,iy=x+dx,y+dy
            if not (0<=ix<nx and 0<=iy<ny):continue
            m=ix*ny+iy
            if not valid[m]:continue
            ng=g+float(np.linalg.norm(pts[m]-pts[n]))
            if ng<cost.get(m,1e9):
                cost[m]=ng;prev[m]=n
                heapq.heappush(q,(ng+float(np.linalg.norm(pts[m]-goal)),ng,m))
    if not found:return None
    route=[np.asarray(goal)]
    n=int(end)
    while n!=start:
        route.append(pts[n]);n=prev[n]
    route.reverse()
    # Remove redundant collinear intermediate points.
    compact=[]
    for p in route:
        if len(compact)>1:
            a=compact[-1]-compact[-2];b=p-compact[-1]
            if abs(a[0]*b[1]-a[1]*b[0])<1e-7 and np.dot(a,b)>=0:compact.pop()
        compact.append(p)
    return compact

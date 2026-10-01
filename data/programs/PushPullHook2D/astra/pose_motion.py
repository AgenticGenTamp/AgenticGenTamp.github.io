"""Collision-aware rigid hook planning in base position and hook angle."""
import heapq
import math
import numpy as np


def path_pose(s, goal, max_expansions=30000):
    s = np.asarray(s, dtype=float)
    goal = np.asarray(goal, dtype=float)
    start = np.array([s[0], s[1], s[11]])
    if goal.shape != (3,) or not np.all(np.isfinite(goal)):
        return None
    def axis(lo, hi, step, a, b):
        return np.unique(np.r_[np.arange(lo, hi + step * .1, step), a, b])
    xs = axis(.105, 3.395, .06, start[0], goal[0])
    ys = axis(.105, 1.145, .05, start[1], goal[1])
    ts = axis(.45, 2.7, .1, start[2], goal[2])
    nx, ny, nt = len(xs), len(ys), len(ts)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    bases = np.stack((xx.ravel(), yy.ravel()), axis=1)
    nxy = nx * ny
    offset = s[9:11] - s[:2]
    button = s[20:22]
    base_valid = np.all((bases >= [.105-1e-7,.105-1e-7]) &
                        (bases <= [3.395+1e-7,1.145+1e-7]), axis=1)
    base_valid &= np.sum((bases-button)**2,axis=1) > .155**2

    def occupancy(angles):
        cs, sn = np.cos(angles), np.sin(angles)
        dc, ds = np.cos(angles-s[11]), np.sin(angles-s[11])
        corners = np.stack((dc*offset[0]-ds*offset[1],
                            ds*offset[0]+dc*offset[1]), axis=1)
        long = -s[18]*np.stack((cs,sn),axis=1)
        short = s[19]*np.stack((sn,-cs),axis=1)
        ends = np.stack((corners,corners+long,corners+short),axis=1)
        lo = .045 - ends.min(axis=1)
        hi = np.array([3.455,2.455])-ends.max(axis=1)
        valid = base_valid[None,:] & np.all(bases[None,:,:]>=lo[:,None,:],axis=2)
        valid &= np.all(bases[None,:,:]<=hi[:,None,:],axis=2)
        rel = button[None,None,:]-bases[None,:,:]-corners[:,None,:]
        for vec in (long,short):
            frac = np.clip(np.sum(rel*vec[:,None,:],axis=2)/np.sum(vec*vec,axis=1)[:,None],0.,1.)
            dist = rel-frac[:,:,None]*vec[:,None,:]
            valid &= np.sum(dist*dist,axis=2)>.115**2
        return valid

    valid = occupancy(ts).reshape(-1)
    # Rotation edges also check halfway through the swept arc.
    mid_valid = occupancy((ts[:-1]+ts[1:])*.5)
    si = (int(np.searchsorted(ts,start[2]))*nx+int(np.searchsorted(xs,start[0])))*ny+int(np.searchsorted(ys,start[1]))
    gi = (int(np.searchsorted(ts,goal[2]))*nx+int(np.searchsorted(xs,goal[0])))*ny+int(np.searchsorted(ys,goal[1]))
    if not valid[gi] and si != gi:
        return None
    valid[si] = True
    hxy = np.linalg.norm(bases-goal[:2],axis=1)
    heuristic = (hxy[None,:]+.5*np.abs(ts[:,None]-goal[2])).reshape(-1)
    costs = np.full(len(valid),np.inf)
    prev = np.full(len(valid),-1,dtype=np.int32)
    costs[si] = 0.
    queue = [(float(heuristic[si]),0.,si)]
    expansions = 0
    found = False
    xy_moves = ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))
    while queue and expansions < max_expansions:
        _, cost, node = heapq.heappop(queue)
        if cost > costs[node]+1e-10:
            continue
        if node == gi:
            found = True
            break
        expansions += 1
        ti, xy = divmod(node,nxy)
        xi, yi = divmod(xy,ny)
        neighbors=[]
        for dx,dy in xy_moves:
            x,y=xi+dx,yi+dy
            if 0<=x<nx and 0<=y<ny:
                neighbors.append(((ti*nx+x)*ny+y,math.hypot(xs[x]-xs[xi],ys[y]-ys[yi])))
        for dt in (-1,1):
            tt=ti+dt
            if 0<=tt<nt and mid_valid[min(ti,tt),xy]:
                neighbors.append((tt*nxy+xy,.5*abs(ts[tt]-ts[ti])))
        for neighbor,increment in neighbors:
            newcost=cost+increment
            if valid[neighbor] and newcost < costs[neighbor]-1e-10:
                costs[neighbor]=newcost
                prev[neighbor]=node
                heapq.heappush(queue,(newcost+float(heuristic[neighbor]),newcost,neighbor))
    if not found:
        return None
    route=[]
    node=gi
    while node!=si:
        ti,xy=divmod(node,nxy);xi,yi=divmod(xy,ny)
        route.append(np.array([xs[xi],ys[yi],ts[ti]]))
        node=int(prev[node])
    route.reverse()
    # Compact collinear grid runs while preserving collision-checked bends.
    compact=[start]
    for p in route:
        if len(compact)>=2:
            a=compact[-1]-compact[-2];b=p-compact[-1]
            if np.linalg.norm(np.cross(a,b))<1e-9 and np.dot(a,b)>0:
                compact.pop()
        compact.append(p)
    return compact[1:]

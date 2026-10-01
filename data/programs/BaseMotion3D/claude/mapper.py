import numpy as np, sys, time, json
from helper import E

RES=0.1
XMIN,XMAX=-3.7,3.1
YMIN,YMAX=-3.9,6.5
def key(i,j): return (i,j)
e=E(0)
free={(0,0):True}
blocked=set()
start=time.time()
# iterative DFS with explicit stack of moves
stack=[((0,0), iter([(1,0),(-1,0),(0,1),(0,-1)]))]
while stack:
    cell, it = stack[-1]
    try:
        d=next(it)
    except StopIteration:
        stack.pop()
        if stack:
            parent=stack[-1][0]
            dx=(parent[0]-cell[0])*RES; dy=(parent[1]-cell[1])*RES
            m,t=e.move(dx,dy)
            assert m, (cell,parent)
        continue
    nb=(cell[0]+d[0], cell[1]+d[1])
    nx,ny=nb[0]*RES, nb[1]*RES
    if nb in free or nb in blocked: continue
    if not (XMIN<=nx<=XMAX and YMIN<=ny<=YMAX):
        blocked.add(nb); continue
    m,t=e.move(d[0]*RES, d[1]*RES)
    if not m:
        blocked.add(nb); continue
    free[nb]=True
    stack.append((nb, iter([(1,0),(-1,0),(0,1),(0,-1)])))
    if len(free)%500==0:
        print(len(free), e.n, time.time()-start, flush=True)
print("free",len(free),"steps",e.n,"time",time.time()-start)
json.dump({"res":RES,"free":sorted(free.keys())}, open("map.json","w"))

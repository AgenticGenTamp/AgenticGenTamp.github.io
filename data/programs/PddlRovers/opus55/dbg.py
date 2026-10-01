from env_client import make_env
from approach import GeneratedApproach
import numpy as np, sys
seed=int(sys.argv[1]); n=int(sys.argv[2]); rn=sys.argv[3]
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(n):
    a=ap.get_action(obs); obs,*_=env.step(a)
for t in range(3):
    a=ap.get_action(obs); print("cmd", ap.cmd[rn], "bad", ap.bad_points[-3:], "tgt", ap.targets[rn]); obs,*_=env.step(a)
S=ap.sides[rn]; x,y=-0.95,-0.45
cur=S.node_of(x,y); tgt=ap.targets[rn][1]; d=S.dist_from_node(tgt)
print("cur",cur,S.pos(cur),"d",d[cur], "tgt", S.pos(tgt))
i,j=S.fi[cur],S.fj[cur]
for di in (-1,0,1):
    print([ (round(d[S.idx[i+di,j+dj]],1) if S.idx[i+di,j+dj]>=0 else None) for dj in (-1,0,1)])
print(ap._point_free(-0.95,-0.45,-1), ap._point_free(-1.0,-0.45,-1))

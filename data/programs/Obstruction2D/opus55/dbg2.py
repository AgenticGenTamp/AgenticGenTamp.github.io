import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed, count, T = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
env = make_env()
obs, info = env.reset(seed=seed, options={'object_count': count})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
for t in range(T):
    a = ap.get_action(obs)
    obs, r, term, trunc, info = env.step(a)
ap._parse(obs)
print('dest', ap.block_dest, 'surface', ap.surface)
for n,b in ap._all_movable().items(): print(n, {k:round(v,3) for k,v in b.items() if k!='name'}, 'hit', ap._column_hit(b))
tgt=ap.block; gx=0.8293184079229832; gy=tgt['y']+tgt['h']+ap._grip_off()+0.0015
carried = (tgt["x"] - gx, tgt["x"] + tgt["w"] - gx, tgt["y"] - gy, tgt["y"] + tgt["h"] - gy)
others=list(ap.obst.values())
B=(ap.block_dest+gx-tgt['x'], 0.112+gy-tgt['y'])
print('B', B)
for o in others+[ap.FLOOR_BOX]:
    print(o['name'], ap._collides(np.array([B]), [o], carried, {}, 0.004), ap._collides(np.array([(gx,gy)]), [o], carried, {}, 0.004))
print(ap._collides(np.array([B]), [], carried, {}, 0.004))
A=(gx,gy)
p=ap._plan_path(A,B,others,carried); print(p)
for h in [0.65,0.7,0.8]:
    path=[A,(A[0],h),(B[0],h),B]
    for i in range(3):
        print(h,i,ap._path_free(path[i:i+2],others,carried,{o['name']:0.0 for o in others+[ap.FLOOR_BOX]},0.004))

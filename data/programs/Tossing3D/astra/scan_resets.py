from env_client import make_env
import numpy as np,json

e=make_env();rows=[]
try:
 for seed in range(200):
  s,i=e.reset(seed=seed,options={'object_count':2});cs=s.get_objects(e.observation_space.get_type('mujoco_movable_object'));cubes=[c for c in cs if c.name.startswith('cube_')]
  ps=[np.array([s.get(c,f) for f in ['x','y','z']]) for c in cubes]
  rows.append([float(np.linalg.norm(ps[0][:2]-ps[1][:2])),seed])
 print(json.dumps(sorted(rows)[:20]),flush=True)
finally:e.close()

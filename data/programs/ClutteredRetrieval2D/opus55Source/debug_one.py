import sys, time, numpy as np
from test_approach import make
import approach
from approach import *
from approach import GeneratedApproach
seed=int(sys.argv[1]); count=int(sys.argv[2])
env=make(); s,info=env.reset(seed=seed, options={'object_count':count})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s,info)
orig=ap._make_plan
def wrapped(state):
    t=time.time()
    try:
        ok=orig(state)
    except Exception as e:
        import traceback; traceback.print_exc(); ok=False
    r,rects=ap._parse(state); m=ap._robot_model(r)
    print(f"plan held={ap._held(r,rects,m)} vac={r['vacuum']} ok={ok} len={len(ap.plan)} dt={time.time()-t:.2f} q={np.round(ap._q(r),3)}")
    return ok
ap._make_plan=wrapped
t0=time.time()
for t in range(300+150*count):
    a=ap.get_action(s); s,r,term,trunc,_=env.step(a)
    if term: print("SUCCESS",t+1); break
    if time.time()-t0>60 or ap.fail_count>5: print("FAIL"); break
r,rects=ap._parse(s); m=ap._robot_model(r)
reg=rects['target_region']; print('region',reg)
for n,d in rects.items():
    if d['type']=='rectangle':
        c=rect_verts(d['x'],d['y'],d['th'],d['w'],d['h']).mean(0)
        rc=rect_verts(reg['x'],reg['y'],reg['th'],reg['w'],reg['h']).mean(0)
        if np.linalg.norm(c-rc)<0.4: print('near region',n,np.round(c,3))
import imageio; imageio.imwrite('logs/final_'+(sys.argv[3] if len(sys.argv)>3 else 'x')+'.png', env.render())

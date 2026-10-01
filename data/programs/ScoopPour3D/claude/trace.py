import sys, time, numpy as np
from env_client import make_env
sys.path.insert(0,'.')
import approach
seed=int(sys.argv[1]); oc=int(sys.argv[2]); MAX=int(sys.argv[3])
env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':oc})
ap=approach.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
GB=obs.get_object_from_name('bin_green_0'); YB=obs.get_object_from_name('bin_yellow_0')
CUBES=[o for o in obs.data if o.name.startswith('cube_')]
def up(q):
    w,x,y,z=q; return np.array([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)])
t0=time.time(); term=False
for i in range(MAX):
    obs,rew,term,trunc,info=env.step(ap.get_action(obs))
    if i%10==0 or term:
        yb=obs.data[YB]; g=obs.data[GB][:3]; c=np.array([obs.data[x][:3] for x in CUBES])
        u=up(yb[3:7]); tilt=np.arctan2(u[1],u[2])
        print(f'{i} ph={ap.phase} st={getattr(ap,"stage",0)} phi={getattr(ap,"phi",0):.2f} a={ap.a:.2f} tilt={tilt:.2f} za={ap.zarm:.3f} ydes={ap.ydes:.3f} zdes={ap.zdes:.3f} yb={np.round(yb[:3],3)} g={np.round(g,3)} cmed={np.round(np.median(c,0),3)} chi={(c[:,2]>g[2]+0.05).sum()}')
    if term or trunc: break
print('TERM',term,'steps',i+1,'wall',round(time.time()-t0,1))
env.close()

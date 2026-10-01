import sys, time, importlib, numpy as np
from env_client import make_env
sys.path.insert(0,'.')
import approach
seed=int(sys.argv[1]); oc=int(sys.argv[2]); MAX=int(sys.argv[3]) if len(sys.argv)>3 else 1000
env=make_env()
obs,info=env.reset(seed=seed, options={'object_count':oc})
ap=approach.GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs,info)
R=obs.get_object_from_name('robot'); GB=obs.get_object_from_name('bin_green_0'); YB=obs.get_object_from_name('bin_yellow_0')
CUBES=[o for o in obs.data if o.name.startswith('cube_')]
t0=time.time(); tot=0
lastphase=None
for i in range(MAX):
    a=ap.get_action(obs)
    obs,rew,term,trunc,info=env.step(a); tot+=rew
    ph=getattr(ap,'phase','?')
    if ph!=lastphase:
        c=np.array([obs.data[x][:3] for x in CUBES])
        print(f'step {i} -> {ph} ybin {np.round(obs.data[YB][:3],3)} green {np.round(obs.data[GB][:3],3)} cube {np.round(np.median(c,0),3)} rew {rew}')
        lastphase=ph
    if i%40==0:
        c=np.array([obs.data[x][:3] for x in CUBES])
        print(f'  {i} ph={ph} roll={getattr(ap,"roll",0):.2f} fx={getattr(ap,"fx",0):.2f} za={getattr(ap,"zarm",0):.2f} cube {np.round(np.median(c,0),3)} green {np.round(obs.data[GB][:3],3)} rew {rew}')
    if term or trunc:
        print('TERMINATED at',i,'term',term,'trunc',trunc); break
c=np.array([obs.data[x][:3] for x in CUBES]); g=obs.data[GB][:3]
d=np.linalg.norm(c-g,axis=1)
print('final rew',rew,'total',round(tot,1),'steps',i+1,'wall',round(time.time()-t0,1))
print('green',np.round(g,3),'cube med',np.round(np.median(c,0),3),'std',np.round(c.std(0),3))
print('dist<5cm',(d<0.05).sum(),'<10',(d<0.10).sum(),'of',len(d),np.round(np.sort(d)[:8],3))
env.close()

import sys, time, numpy as np
from env_client import make_env
sys.path.insert(0,'.')
import approach
seed=int(sys.argv[1]); oc=int(sys.argv[2]); MAX=int(sys.argv[3]) if len(sys.argv)>3 else 600
env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':oc})
ap=approach.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
GB=obs.get_object_from_name('bin_green_0'); CUBES=[o for o in obs.data if o.name.startswith('cube_')]
t0=time.time(); term=False; i=0
for i in range(MAX):
    obs,rew,term,trunc,info=env.step(ap.get_action(obs))
    if term or trunc: break
c=np.array([obs.data[x][:3] for x in CUBES]); g=obs.data[GB][:3]
print(f'seed={seed} oc={oc} TERM={term} steps={i+1} wall={time.time()-t0:.1f} green={np.round(g,3)} cubemed={np.round(np.median(c,0),3)} zmin={c[:,2].min():.3f} phase={ap.phase}')
env.close()

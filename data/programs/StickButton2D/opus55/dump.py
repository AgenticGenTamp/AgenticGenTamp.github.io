from env_client import make_env
import sys
env=make_env()
tf=env.observation_space.type_features
def dump(obs):
    for n in sorted(obs.get_object_names()):
        o=obs.get_object_from_name(n)
        print(' ',n,o.type.name,[ (f,round(float(obs.get(o,f)),3)) for f in tf[o.type] ])
if __name__=='__main__':
  for seed in range(3):
    obs,info=env.reset(seed=seed)
    print('seed',seed,info)
    dump(obs)

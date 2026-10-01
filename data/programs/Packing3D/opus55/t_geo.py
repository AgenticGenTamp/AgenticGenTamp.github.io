from env_client import make_env
import approach
env=make_env(); obs,info=env.reset(seed=3)
approach.set_geometry(obs)
print(approach.RACK, approach.FLOOR, approach.PLACE_Z, approach.ZOFF)

from grasp_utils import *
env = new_env()
obs, info = env.reset(seed=0)
for n in sorted(obs.get_object_names()):
    if n.startswith('block'): print(n, block_corners(blk(obs,n)).round(3).tolist())

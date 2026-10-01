from env_client import make_env
import numpy as np
env = make_env()
for seed in range(8):
    obs, info = env.reset(seed=seed)
    R=obs.get_object_from_name('robot')
    s=f"seed{seed} n={info['object_count']} base=({obs.get(R,'pos_base_x'):.3f},{obs.get(R,'pos_base_y'):.3f},{obs.get(R,'pos_base_rot'):.3f}) "
    for n in ['bin_green_0','bin_yellow_0','scoop_0']:
        o=obs.get_object_from_name(n); s+=f"{n}=({obs.get(o,'x'):.3f},{obs.get(o,'y'):.3f},{obs.get(o,'z'):.3f},qz={obs.get(o,'qz'):.2f}) "
    cs=[o for o in obs.get_objects(env.observation_space.get_type('mujoco_movable_object')) if o.name.startswith('cube')]
    P=np.array([[obs.get(o,'x'),obs.get(o,'y'),obs.get(o,'z')] for o in cs])
    s+=f"cubes min{P.min(0).round(3)} max{P.max(0).round(3)}"
    print(s)

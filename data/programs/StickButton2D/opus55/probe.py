from env_client import make_env
env=make_env()
for seed in range(4):
    obs,info=env.reset(seed=seed)
    for n in obs.get_object_names():
        o=obs.get_object_from_name(n)
        print(seed,n,{f:round(obs.get(o,f),3) for f in (['x','y','theta','base_radius','arm_joint','arm_length','vacuum','gripper_height','gripper_width'] if n=='robot' else ['x','y','theta','static','width','height','radius'] ) if True} if n=='robot' else None)

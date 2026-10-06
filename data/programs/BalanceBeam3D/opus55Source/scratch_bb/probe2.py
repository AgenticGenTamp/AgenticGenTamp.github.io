from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env=KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0")
md=env._kinder_env.metadata
for k in ['observation_space_description','action_space_description','reward_description']:
    print("=== ",k); print(md.get(k))

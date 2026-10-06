from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv",count_kwarg="num_buttons",count_object_prefix="button",bilevel_env_name="stickbutton2d",design_counts=[3],eval_counts=[3])
for sd in range(3000):
    s,_=env.reset(seed=sd,options={'object_count':3})
    st=s.get_object_from_name('stick'); x=s.get(st,'x')+0.025
    if x<0.06 or x>3.44: print(sd, round(x,3))

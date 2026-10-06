import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
np.set_printoptions(precision=4, suppress=True, linewidth=150)
env = VariableObjectCountEnv(
    constant_object_env_path="kinder.envs.dynamic3d.task_families:SweepSimple3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube_",
    constant_object_env_kwargs={'scene_bg': False},
    design_counts=[1,5], eval_counts=[1,5,10])
state, info = env.reset(seed=0, options={'object_count': 5})
print("info keys:", list(info.keys()) if isinstance(info, dict) else info)
for name in sorted(state.get_object_names()):
    obj = state.get_object_from_name(name)
    feats = env.type_features[obj.type]
    print(name, obj.type.name, {f: round(float(state.get(obj, f)), 4) for f in feats})
print("action_space:", env.action_space)
print("low:", env.action_space.low); print("high:", env.action_space.high)
b = env._current_backend
print("backend:", type(b), "max_episode_steps:", getattr(b, 'spec', None))
oc = b._object_centric_env
print(type(oc).__mro__)
for attr in ['_max_episode_steps','max_episode_steps']:
    print(attr, getattr(oc, attr, None), getattr(b, attr, None))
s, r, term, trunc, inf = env.step(env.action_space.sample()*0 + np.clip(0, env.action_space.low, env.action_space.high))
print("step r", r, term, trunc)
print("max_steps_for_count(5):", env.max_steps_for_count(5))
spec = getattr(b, 'spec', None); print("spec", spec)
print("cfg:", getattr(oc, 'config', None) or getattr(oc, '_config', None))
r_list=[]
st=state
for t in range(3):
    st, r, term, trunc, inf = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    r_list.append(r)
rob = st.get_object_from_name('robot')
print("after 3 zero-actions rewards", r_list, {f: round(float(st.get(rob,f)),3) for f in env.type_features[rob.type][:11]})
# hold-in-place action
s0=state; rob0=s0.get_object_from_name('robot')
hold=np.array([s0.get(rob0,f) for f in env.type_features[rob0.type][:11]],dtype=np.float32)
print("hold action:", hold)

exec(open('look2.py').read().split("print(type(s2)")[0])
for n in ['robot','cube0']:
    o1=s1.get_object_from_name(n); o2=s2.get_object_from_name(n)
    print(n, [round(float(s1.get(o1,f)),4) for f in env.type_features[o1.type]][:10])
    print(n, [round(float(s2.get(o2,f)),4) for f in env.type_features[o2.type]][:10])
print(env._max_episode_steps if hasattr(env,'_max_episode_steps') else None, [k for k in vars(env).keys()])
import inspect, kinder.envs.kinematic3d.transport3d as T
print(inspect.signature(T.ObjectCentricTransport3DEnv.__init__))

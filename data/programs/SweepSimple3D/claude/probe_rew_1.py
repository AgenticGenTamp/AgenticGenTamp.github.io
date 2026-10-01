from env_client import make_env
import numpy as np
env=make_env(); os_=env.observation_space
T=os_.get_type
print("ENV DIR:", [d for d in dir(env) if not d.startswith('_')])
print("ENV DICT:", list(vars(env).keys()))
for seed in [0,1,2,3]:
    obs,info=env.reset(seed=seed)
    print("=== seed",seed,"reset info keys:",list(info.keys()) if isinstance(info,dict) else info)
    for tn in ["mujoco_movable_object","mujoco_fixture","mujoco_tidybot_robot"]:
        try: objs=obs.get_objects(T(tn))
        except Exception as e: print(tn,"ERR",e); continue
        for o in objs:
            fs=os_.type_features[T(tn)]
            if tn=="mujoco_tidybot_robot":
                print(" ",o.name,{f:round(float(obs.get(o,f)),3) for f in fs[:3]})
            else:
                print(" ",o.name,{f:round(float(obs.get(o,f)),3) for f in fs if f in ("x","y","z","bb_x","bb_y","bb_z")})
    z=np.zeros(11,dtype=np.float32)
    rs=[]
    for i in range(20):
        obs,r,te,tr,info=env.step(z); rs.append(round(float(r),5))
    print(" rewards:",rs)
    print(" step info:",info)
env.close()

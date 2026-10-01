import sys, json, numpy as np
from env_client import make_env

def dump(seed):
    env = make_env()
    obs, info = env.reset(seed=seed)
    rec = {"seed": seed, "info": str(info)}
    objs = obs.get_object_names()
    rec["names"] = sorted(objs)
    d = {}
    for n in sorted(objs):
        o = obs.get_object_from_name(n)
        feats = obs.type_features[o.type]
        d[n] = {"type": str(o.type.name if hasattr(o.type,'name') else o.type),
                **{f: round(float(v),5) for f, v in zip(feats, obs.data[o])}}
    rec["objects"] = d
    # zero action step reward
    a = np.zeros(11, dtype=np.float32)
    o2, r, term, trunc, info2 = env.step(a)
    rec["r_zero_action"] = float(r)
    rec["info_step"] = str(info2)
    env.close()
    return rec

if __name__ == "__main__":
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    out = [dump(s) for s in range(lo, hi)]
    print(json.dumps(out))

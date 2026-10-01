import numpy as np, json
from env_client import make_env
env = make_env()
rows=[]
infos=set()
for s in range(30):
    obs, info = env.reset(seed=s)
    o=np.asarray(obs,dtype=float)
    infos.add(json.dumps({k:str(type(v)) for k,v in (info or {}).items()},sort_keys=True))
    rows.append(o.copy())
    if s==0:
        print("info@reset:", info)
R=np.array(rows)
def rng(name, idx):
    sub=R[:,idx]
    print(name, "min",np.round(sub.min(0),4), "max",np.round(sub.max(0),4), "std", np.round(sub.std(0),4))
rng("large_block xyz", slice(0,3))
rng("large_quat", slice(3,7))
rng("large_bbox", slice(13,16))
rng("robot base", slice(16,19))
rng("arm joints", slice(19,27))
rng("seesaw xyz", slice(38,41))
rng("seesaw quat", slice(41,45))
rng("seesaw bbox", slice(51,54))
rng("sb1 xyz", slice(54,57))
rng("sb1 quat", slice(57,61))
rng("sb1 bbox", slice(67,70))
rng("sb2 xyz", slice(70,73))
rng("sb2 quat", slice(73,77))
rng("sb2 bbox", slice(83,86))
print("info type sets:", infos)
np.save("seed_obs.npy", R)
print("first3 seeds full-ish:")
for i in range(3):
    o=R[i]
    print(i,"lb",np.round(o[0:3],3),"ss",np.round(o[38:41],3),np.round(o[41:45],3),"bb",np.round(o[51:54],3),"sb1",np.round(o[54:57],3),"sb2",np.round(o[70:73],3))

from env_client import make_env
import numpy as np
env = make_env()
def d(obs,n):
    o=obs.get_object_from_name(n); return {f: round(float(obs.get(o,f)),5) for f in obs.type_features[o.type]}
obs,info=env.reset(seed=42)
tb=d(obs,'target_block'); cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
print("block top",top)
for i in range(40):
    r=d(obs,'robot'); dx=np.clip(cx-r['x'],-0.05,0.05)
    if abs(dx)<1e-5: break
    obs,*_=env.step(np.array([dx,0,0,0,0]))
prev=None
for i in range(200):
    obs,*_=env.step(np.array([0,-0.002,0,0,0]))
    r=d(obs,'robot')
    if prev is not None and abs(r['y']-prev)<1e-7:
        break
    prev=r['y']
r=d(obs,'robot'); print("contact y",r['y'],"arm",r['arm_joint'],"offset",r['y']-top)
# now try arm retract fine
for i in range(20):
    obs,*_=env.step(np.array([0,0,0,-0.01,0]))
print("after retract",d(obs,'robot')['arm_joint'])
# extend fine until blocked
prev=None
for i in range(100):
    obs,*_=env.step(np.array([0,0,0,0.002,0]))
    r=d(obs,'robot')
    if prev is not None and abs(r['arm_joint']-prev)<1e-7: break
    prev=r['arm_joint']
print("arm blocked at",d(obs,'robot')['arm_joint'], "robot y", d(obs,'robot')['y'])
# vacuum on
obs,*_=env.step(np.array([0,0,0,0,1.0]))
print("after vac", d(obs,'robot'), d(obs,'target_block'))
# move up
for i in range(4): obs,*_=env.step(np.array([0,0.05,0,0,1.0]))
print("lifted", d(obs,'robot')['y'], d(obs,'target_block'))
for i in range(4): obs,*_=env.step(np.array([-0.05,0,0,0,1.0]))
print("left", d(obs,'robot')['x'], d(obs,'target_block'))
# rotate
for i in range(2): obs,*_=env.step(np.array([0,0,0.19,0,1.0]))
print("rot", d(obs,'robot'), d(obs,'target_block'))
# release
obs,rew,t,tr,info=env.step(np.array([0,0,0,0,0.0]))
print("release", d(obs,'target_block'), t)
for i in range(10): obs,rew,t,tr,info=env.step(np.array([0,0,0,0,0.0]))
print("after wait", d(obs,'target_block'), d(obs,'robot'))
env.close()

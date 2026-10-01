from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
R = obs.get_object_from_name('robot')
feats = obs.type_features[R.type]
def rv(o): return np.array([obs.get(R,f) for f in feats])
def show(tag):
    print(tag, {f: round(float(obs.get(R,f)),4) for f in feats if not f.startswith('grasp_tf')})
show('init')
a = np.zeros(11,dtype=np.float32)
obs,r,te,tr,_ = env.step(a); show('zero'); print(r,te,tr)
a[3]=0.5; obs,*_ = env.step(a); show('j1 +0.5 (clip?)')
a[:]=0; a[0]=0.1; obs,*_ = env.step(a); show('base x+0.1')
a[:]=0; a[2]=0.1; obs,*_ = env.step(a); show('base rot+0.1')
a[:]=0; a[10]=-1; obs,*_ = env.step(a); show('close')
obs,*_ = env.step(a); show('close2')
a[:]=0; a[10]=1; obs,*_ = env.step(a); show('open')
a[:]=0; a[5]=0.2; 
for i in range(20): obs,*_ = env.step(a)
show('j3 +4.0')
a[:]=0; a[4]=0.2
for i in range(20): obs,*_ = env.step(a)
show('j2 +4.0')

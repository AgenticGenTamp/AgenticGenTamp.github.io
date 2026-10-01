from env_client import make_env
import numpy as np,json
from kinematics_candidate import forward
np.set_printoptions(precision=4,suppress=True)
e=make_env()
for seed in [51,55]:
 s,_=e.reset(seed=seed);print(seed,s[:48].reshape(3,16))
 json.dump(s.tolist(),open('initial_%d.json'%seed,'w'))
e.close()
q=np.array([0,.96291351,np.pi,-.85752203,0,-1.32115712,np.pi/2]);print('RDEFAULT',forward(q)[1]);q[-1]=0;print('RZERO',forward(q)[1])

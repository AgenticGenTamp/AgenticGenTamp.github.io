import time, json
from expA_lib import *
t=time.time(); env,obs,blk=new_rig(); print("setup",round(time.time()-t,1),blk)
for (a,b,c) in [(0.03,0,0),(0.03,0,0),(0.0,0,0)]:
    t=time.time(); obs,r=trial(env,obs,blk,a,b,c); print(json.dumps(r), round(time.time()-t,1))
env.close()

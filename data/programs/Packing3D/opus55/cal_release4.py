from cal_trial import *
t=trial(0,'part0',0,0,0.05); d=t['d']
for g in [1.0]*3:
    o,r,te,tr,info=d.env.step(act(grip=g)); d.obs=o
    R=o.get_object_from_name('robot'); print(g,r,te,tr,info,o.get(R,'grasp_active'),o.get(R,'finger_state'))
# try small arm move combined with open
o,r,te,tr,info=d.env.step(act(dq=np.r_[0,0.01,0,0,0,0,0],grip=1.0)); d.obs=o; print('move+open',d.r['ga'],d.r['q'][1])

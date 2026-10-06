import numpy as np, time
exec(open('approach.py').read()); exec(open('slp.py').read())
s=Sim(1); st=s.get_state(); P=Planner(s)
s.ik((0,0,1),Q0,np.array(st[:10]),range(3,10))  # init ff
qc=np.array(st[:10]); 
for c in [(-0.1,0.0,0.781),(0.1,0.3,0.781),(-0.2,-0.3,0.781)]:
    for yaw in [0,0.5,1.0]:
        t=time.time()
        q=s.ik(np.array(c)+[0,0,0.03], top_quat(yaw), qc, list(range(10)), weights=[1.5]*3+[1]*7)
        t1=time.time()-t
        if q is None: print('ik fail'); continue
        t=time.time()
        r=slp_solve(s, qc, q, TaskGrasp(c,yaw), base_region(qc))
        r0=slp_solve(s, qc, qc, TaskGrasp(c,yaw), base_region(qc))
        print(c,yaw,'ik L∞ %.3f'%np.max(np.abs(conf_delta(qc,q))), 'slp', None if r is None else round(r[1],3), 'slp from qc', None if r0 is None else round(r0[1],3), 'time %.3f %.3f'%(t1,time.time()-t))

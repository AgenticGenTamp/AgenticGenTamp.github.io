from env_client import make_env
import numpy as np, math
def d(obs,n):
    o=obs.get_object_from_name(n); return {f: float(obs.get(o,f)) for f in obs.type_features[o.type]}
def mk():
    env=make_env(); obs,info=env.reset(seed=42)
    st={'obs':obs}
    def step(a):
        o,*_=env.step(np.array(a,dtype=float)); st['obs']=o; return d(o,'robot')
    R=lambda: d(st['obs'],'robot')
    def rot_to(th):
        for i in range(400):
            e=th-R()['theta']
            while e>math.pi: e-=2*math.pi
            while e<-math.pi: e+=2*math.pi
            if abs(e)<1e-7: break
            step([0,0,float(np.clip(e,-0.196,0.196)),0,0])
    def goto(axis,t):
        for i in range(900):
            v=R()['x'] if axis==0 else R()['y']; e=t-v
            if abs(e)<1e-7: break
            a=[0,0,0,0,0]; a[axis]=float(np.clip(e,-0.049,0.049)); step(a)
    def setarm(t):
        for mag in (0.05,0.005,0.0005):
            for i in range(400):
                e=t-R()['arm_joint']
                if abs(e)<1e-7: break
                p=R()['arm_joint']; step([0,0,0,float(np.clip(e,-mag,mag)),0])
                if abs(R()['arm_joint']-p)<1e-9: break
    def push(axis,sign):
        for mag in (0.049,0.005,0.0005,0.00005,0.00001):
            prev=None
            for i in range(1500):
                a=[0,0,0,0,0]; a[axis]=sign*mag
                r=step(a); v=r['x'] if axis==0 else r['y']
                if prev is not None and abs(v-prev)<1e-9: break
                prev=v
        r=R(); return round(r['x'] if axis==0 else r['y'],6)
    return env,st,step,R,rot_to,goto,setarm,push

# side walls with arm extended horizontally
for th,sgn,lab in [(0.0,+1,'theta=0 arm+x'),(math.pi,-1,'theta=pi arm-x')]:
    env,st,step,R,rot_to,goto,setarm,push=mk()
    goto(1,0.5); goto(0,0.8); rot_to(th); setarm(0.2)
    lim=push(0,sgn)
    print("%-16s arm=%.4f  x_limit=%s -> wall at %.4f"%(lab,R()['arm_joint'],lim,lim+sgn*(R()['arm_joint']+0.005)))
    env.close()
# base-only walls (arm vertical)
env,st,step,R,rot_to,goto,setarm,push=mk()
goto(1,0.5); rot_to(math.pi/2); setarm(0.1)
print("arm UP: minx=%s maxx=%s"%(push(0,-1),push(0,+1)))
env.close()

# Q5 rotation: grasp block and rotate
env,st,step,R,rot_to,goto,setarm,push=mk()
tb=d(st['obs'],'target_block'); cx=tb['x']+tb['width']/2; top=tb['y']+tb['height']
goto(1,0.6); goto(0,cx); rot_to(-math.pi/2); setarm(0.1)
push(1,-1); setarm(0.2); push(1,-1)
print("pre-grasp robot",{k:round(v,4) for k,v in R().items() if k in('x','y','theta','arm_joint')})
step([0,0,0,0,1.0])
print("block after vac",{k:round(d(st['obs'],'target_block')[k],4) for k in('x','y','theta')})
for i in range(3): step([0,0.049,0,0,1.0])
b0=d(st['obs'],'target_block'); r0=R()
print("lifted robot",round(r0['x'],4),round(r0['y'],4),"block",round(b0['x'],4),round(b0['y'],4),round(b0['theta'],4))
for i in range(4):
    step([0,0,0.196,0,1.0])
    b=d(st['obs'],'target_block'); r=R()
    print("  rot%d robot(x,y,th)=(%.5f,%.5f,%.4f) block(x,y,th)=(%.4f,%.4f,%.4f)"%(i,r['x'],r['y'],r['theta'],b['x'],b['y'],b['theta']))
env.close()

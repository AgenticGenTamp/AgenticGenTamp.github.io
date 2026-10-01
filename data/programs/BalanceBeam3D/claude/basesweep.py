import numpy as np, sys
import kinova, ctrl
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
AX=int(sys.argv[1]); SGN=float(sys.argv[2]); Z=float(sys.argv[3]); G=float(sys.argv[4])
env=make_env(); obs,_=env.reset(seed=0)
def drive(bx,by,byaw,n=100,g=0.0):
    global obs
    for i in range(n):
        a=np.zeros(11)
        a[0]=np.clip((bx-obs[16])/ctrl.BASE_GAIN,-0.1,0.1)
        a[1]=np.clip((by-obs[17])/ctrl.BASE_GAIN,-0.1,0.1)
        a[2]=np.clip((byaw-obs[18])/ctrl.YAW_GAIN,-0.1,0.1)
        a[10]=g
        obs,r,te,tr,_=env.step(a.astype(np.float32))
        if abs(bx-obs[16])<5e-4 and abs(by-obs[17])<5e-4: break
def servo_rel(rel,n,grip,tol=None):
    global obs
    for i in range(n):
        pt=np.array([obs[16]+rel[0],obs[17]+rel[1],rel[2]])
        dq=kinova.ik_step(obs[19:26], ctrl.w2a(pt,obs[16:19]), ctrl.Rdown, tool_offset=ctrl.TOOL)
        a=np.zeros(11); a[3:10]=np.clip(dq/ctrl.ARM_GAIN,-0.1,0.1); a[10]=grip
        obs,r,te,tr,_=env.step(a.astype(np.float32))
        if tol is not None and np.linalg.norm(ctrl.ee_world(obs)-pt)<tol and i>3: break
    return ctrl.ee_world(obs)
blk=obs[54:57].copy(); print("blk",blk)
# park base far, well away from all blocks (behind and offset in y)
b0=np.array([blk[0]-0.45,blk[1],0.0]); b0[AX]-=SGN*0.15
drive(b0[0],b0[1],0.0)
print("ee after park hi:",servo_rel([0.45,0.0,0.30],300,G,tol=0.004), obs[54:57])
print("ee after park lo:",servo_rel([0.45,0.0,Z],300,G,tol=0.003), obs[54:57])
ref=obs[54:57].copy()
tgt=np.array([obs[16],obs[17]])
for k in range(40):
    tgt[AX]+=SGN*0.006
    drive(tgt[0],tgt[1],0.0,n=30,g=G)
    servo_rel([0.45,0.0,Z],12,G)
    ee=ctrl.ee_world(obs); b=obs[54:57]
    d=np.linalg.norm(b[:2]-ref[:2])
    print(f"  ee={ee} blk={b} d={d:.4f}")
    if d>0.006:
        print("CONTACT ee=",ee,"blk_ref",ref); break
env.close()

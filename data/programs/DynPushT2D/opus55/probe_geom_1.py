from env_client import make_env
import numpy as np, sys
np.set_printoptions(precision=4, suppress=True)
env = make_env()
def R(t): return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
def step(a):
    global obs
    obs,r,te,tr,info = env.step(np.clip(a,-0.0499,0.0499)); return obs
def goto(tgt, tol=1e-4, n=400):
    for i in range(n):
        d = tgt-obs[16:18]
        if np.linalg.norm(d)<tol: break
        step(d)
def route(tgt, cen, Rr):
    # go radially out to Rr, arc, then go to tgt
    p = obs[16:18]; v=p-cen; a0=np.arctan2(v[1],v[0])
    goto(cen+Rr*np.array([np.cos(a0),np.sin(a0)]))
    v=tgt-cen; a1=np.arctan2(v[1],v[0])
    da=(a1-a0+np.pi)%(2*np.pi)-np.pi
    for k in range(1,41):
        a=a0+da*k/40; goto(cen+Rr*np.array([np.cos(a),np.sin(a)]),tol=1e-3)
    goto(tgt)
def probe(seed, start_local, dir_local, name):
    global obs
    obs,_=env.reset(seed=seed)
    pose=obs[0:3].copy(); w,lh,lv=obs[12],obs[13],obs[14]
    Rm=R(pose[2]); cen=pose[:2]+Rm@np.array([0,-lv/2])
    start=pose[:2]+Rm@np.array(start_local)
    route(start,cen,1.05)
    assert np.allclose(obs[0:3],pose), ('moved en route',obs[0:3],pose)
    d=Rm@np.array(dir_local); d/=np.linalg.norm(d)
    prev=obs[16:18].copy()
    for i in range(2000):
        step(d*0.001)
        if not np.allclose(obs[0:3],pose,atol=1e-7):
            loc_prev=Rm.T@(prev-pose[:2]); loc_now=Rm.T@(obs[16:18]-pose[:2])
            print(f"seed{seed} {name}: w={w:.4f} lh={lh:.4f} lv={lv:.4f} contact robot local prev={loc_prev} now={loc_now} dpose={obs[0:3]-pose}")
            return
        prev=obs[16:18].copy()
    print(name,'no contact')
for seed in [int(s) for s in sys.argv[1:]]:
    obs,_=env.reset(seed=seed); w,lh,lv=obs[12],obs[13],obs[14]
    print('pose',obs[0:3],'robot',obs[16:18],'goal',obs[29:32])
    if 0: probe(seed,[0, 0.4],[0,-1],'bar top (x=0)')
    probe(seed,[0.3, 0.4],[0,-1],'bar top (x=0.3)')
    probe(seed,[0,-lv-0.3],[0,1],'stem bottom')
    probe(seed,[lh/2+0.3,0],[-1,0],'bar end +x (y=0)')
    probe(seed,[-lh/2-0.3,0],[1,0],'bar end -x (y=0)')
    probe(seed,[0.35,-lv/2],[-1,0],'stem side +x mid')
    probe(seed,[-0.35,-lv/2],[1,0],'stem side -x mid')
    probe(seed,[0.4,-0.45],[0,1],'bar underside x=0.4')

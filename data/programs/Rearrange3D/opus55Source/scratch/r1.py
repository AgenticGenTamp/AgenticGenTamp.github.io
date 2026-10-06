import sys; sys.path.insert(0,'/sandbox/scratch'); from h import *
oc=env._kinder_env._object_centric_env; bowlo=oc._objects_dict['bowl_0']
def inreg(p): return bowlo.check_in_region(np.array(p,dtype=np.float32),'bowl_0_region',re)
def body(m,n): return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
def tilt(h): b=body(h.m,"bowl_0"); return float(np.degrees(np.arccos(np.clip(h.d.xmat[b][8],-1,1))))
OPEN=8; RET=np.array([0,0,0.10])
def run(seed, ox, oy, gang, zdrop, settle0=60, verbose=False):
    h=H(seed); h.step(settle0)
    dp=h.drink(); bp=h.obs[0:3].copy()
    dv=np.array([np.cos(gang),np.sin(gang)]); g=bp[:2]+0.058*dv; yaw=gang
    T=dp[:2]+np.array([ox,oy])
    h.setbase([-0.17, np.clip(0.5*(bp[1]+T[1]),-0.6,0.6), 0])
    top=bp[2]+0.0184
    h.move([g[0],g[1],top+0.07], yaw, 0.025); h.move([g[0],g[1],top-0.012], yaw, 0.01); h.close(8)
    h.move([g[0],g[1],0.62], yaw, 0.015)
    p,_=h.pinch(); off=p-h.obs[0:3]
    h.move([T[0]+off[0],T[1]+off[1],0.62], yaw, 0.025)
    # lower: pinch target z = zdrop
    h.move([T[0]+off[0],T[1]+off[1],zdrop], yaw, 0.006, settle=15)
    if verbose: print(" before open bowl",np.round(h.obs[0:3],3),"tilt",round(tilt(h),1))
    h.open(OPEN)
    if verbose: print(" after open bowl",np.round(h.obs[0:3],3),"tilt",round(tilt(h),1))
    p,_=h.pinch(); h.move(p+RET, yaw, 0.01, settle=5)
    if verbose: print(" after retract bowl",np.round(h.obs[0:3],3),"tilt",round(tilt(h),1))
    for k in range(15):
        h.step(10)
        if verbose: print('   ',k*10,np.round(h.obs[0:3],3),round(tilt(h),1), np.round(h.obs[7:13],2))
    dr=h.drink(); print(f"seed{seed} off({ox:+.3f},{oy:+.3f}) gang{np.degrees(gang):.0f} zdrop{zdrop:.3f} -> bowl {np.round(h.obs[0:3],3)} tilt {tilt(h):.1f} drink {np.round(dr,3)} IN {inreg(dr)}", flush=True)
    return h
if __name__=="__main__":
    if len(sys.argv)>4: exec(sys.argv[4])
    seed=int(sys.argv[1])
    for args in eval(sys.argv[2]): run(seed,*args, settle0=int(sys.argv[3]) if len(sys.argv)>3 else 60, verbose=True)

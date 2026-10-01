import numpy as np, sys, json
from env_client import make_env
from ctrl import move
np.set_printoptions(precision=4,suppress=True,linewidth=250)
X=float(sys.argv[1]); Y=float(sys.argv[2]); Z0=float(sys.argv[3]); Z1=float(sys.argv[4]); seed=int(sys.argv[5]) if len(sys.argv)>5 else 0
env=make_env(); o,_=env.reset(seed=seed); o=np.asarray(o,float)
c0=o[:80].reshape(5,16)[:,:3].copy(); w0=o[147:150].copy()
o,e,u=move(env,o,[X,Y,Z0],steps=250)
print("approach err",round(e,4),"steps",u,"base",np.round(o[125:128],4))
tot=u
for z in np.arange(Z0,Z1-1e-9,-0.02):
    o,e,u=move(env,o,[X,Y,z],steps=60); tot+=u
    c=o[:80].reshape(5,16)[:,:3]; d=np.linalg.norm(c-c0,axis=1); dw=np.linalg.norm(o[147:150]-w0)
    print(f"z={z:+.3f} jerr={e:.4f} u={u} tot={tot} cubes={np.round(d,3)} wiper={dw:.3f}")
    if d.max()>0.008 or dw>0.008:
        print("CONTACT cube_idx",int(np.argmax(d)),"cube_now",np.round(c[np.argmax(d)],4),"c0",np.round(c0[np.argmax(d)],4),"wiper_now",np.round(o[147:150],4),"w0",np.round(w0,4))
        break
    if tot>800: break
print("final joints",np.round(o[128:135],4))
env.close()

import numpy as np, sys
from env_client import make_env
from ctrl import move
from fk import fk
np.set_printoptions(precision=4,suppress=True,linewidth=250)
X=float(sys.argv[1]); Y=float(sys.argv[2])
env=make_env(); o,_=env.reset(seed=0); o=np.asarray(o,float)
c0=o[:80].reshape(5,16)[:,:3].copy(); w0=o[147:150].copy()
o,e,u=move(env,o,[X,Y,0.35],steps=250); tot=u
print(f"# X={X} Y={Y} approach jerr={e:.4f} u={u}")
for z in np.arange(0.34,-0.25,-0.02):
    o,e,u=move(env,o,[X,Y,z],steps=80); tot+=u
    c=o[:80].reshape(5,16)[:,:3]; d=np.linalg.norm(c-c0,axis=1); dw=np.linalg.norm(o[147:150]-w0)
    p=fk(o[128:135])[:3,3]
    print(f"z={z:+.3f} jerr={e:.4f} fkz={p[2]:+.4f} u={u} tot={tot} dc={np.round(d,3)} dw={dw:.3f}")
    if tot>900: break
print("q_end",np.round(o[128:135],4),"base",np.round(o[125:128],4))
env.close()

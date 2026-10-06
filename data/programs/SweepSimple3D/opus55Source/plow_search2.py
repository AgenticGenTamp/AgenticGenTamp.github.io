from plow_search import *
for ang in range(0,360,45):
    a=np.deg2rad(ang); app=(np.cos(a),np.sin(a),0)
    for close in ['v','h']:
        c=(0,0,1) if close=='v' else (-np.sin(a),np.cos(a),0)
        R=rot(app,c)
        for xp in [0.3,0.45,0.6]:
            for yp in [-0.3,-0.15,0,0.15,0.3]:
                zp=0.075 if close=='v' else 0.05
                q,ep,er=ikn(np.array([xp,yp,zp]),R,q_init,q_init,iters=400)
                if ep>2e-3 or er>2e-3: continue
                pr=profile(q)
                if pr is None: continue
                zmin,x0,x1,y0,y1=pr
                print(ang,close,xp,yp,zp,'zmin %.3f low x[%.2f,%.2f] y[%.2f,%.2f] maxdq %.2f'%(zmin,x0,x1,y0,y1,np.max(np.abs(q-q_init))), np.round(q-q_init,2).tolist(), flush=True)

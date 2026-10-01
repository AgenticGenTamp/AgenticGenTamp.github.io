exec(open('probe_contact.py').read().split("for trial in range(30):")[0])
rng=np.random.default_rng(4); moved=0
for trial in range(60):
 target=home+rng.uniform([-1,-.8,-.5,-.8,-.7,-.7,-1],[1,.8,.5,.8,.7,.7,1])
 for k in range(25):
  a=np.zeros(11,np.float32);a[3:10]=np.clip((target-o[128:135])*.5,-.1,.1);a[10]=1;o,*_=e.step(a)
 d=np.linalg.norm(o[147:150]-wp)
 if d>moved+.003: moved=d;print('move',trial,round(d,3),np.round(o[147:150],3),np.round(target,2))
print('final',o[147:150],o[:80].reshape(5,16)[:,:3]);e.close()

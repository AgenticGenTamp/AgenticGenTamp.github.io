s=open('approach.py').read()
s=s.replace("self.side=(-1 if cur[0]<0 else 1) if abs(pos[0])<.08 else (-1 if pos[0]<0 else 1)","self.side=-1")
a=s.index('    high=self.solve(base,[*pos[:2],1.04],yaw)')
b=s.index("    self.stage='lift'",a)
s=s[:a]+'''    if pos[0]-base[0]>.73:
     rot=Rotation.from_euler('y',.7).as_matrix()
     tool=pos-.04*rot[:,0];q=Q0
     for z in [1.04,.98,.92,.87,tool[2]]:
      p=tool-np.r_[base[:2],0.];p[2]=z
      q=ik(p,q0=q,rot=rot);self.add(np.r_[base,q],1)
     self.add(np.r_[base,q],-1)
    else:
     high=self.solve(base,[*pos[:2],1.04],yaw)
     self.add(high,1)
     mid=high
     for z in [1.0,.94,.88]:
      mid=self.solve(base,[*pos[:2],z],yaw,q0=mid[3:]);self.add(mid,1)
     low=self.solve(base,[*pos[:2],pos[2]+.038],yaw,q0=mid[3:])
     self.add(low,1);self.add(low,-1)
''' +s[b:]
open('candidate_tilt_policy.py','w').write(s)

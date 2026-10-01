"""Helper: fetch the hook and hold it as a pan. Uses lib2.Ctl."""
import numpy as np
from lib2 import Ctl
PI=np.pi

def wrap(a): return (a+PI)%(2*PI)-PI

class Scoop(Ctl):
    def fetch_hook(self):
        hx=self.g('x',self.H); hy=self.g('y',self.H)
        top=hy+self.g('length_side1',self.H)
        self.goto(x=hx,y=2.0,th=-PI/2,arm=0.2,gap=0.25,tol=0.01)
        self.goto(y=top+self.g('arm_joint')+0.02)
        for _ in range(30):
            self.step([0,0,0,0,-0.015])
            if self.g('held',self.H)>0.5: break
        self.rel()
        return self.g('held',self.H)>0.5
    def rel(self):
        th=self.g('theta'); d=np.array([self.g('x',self.H)-self.g('x'), self.g('y',self.H)-self.g('y')])
        c,s=np.cos(th),np.sin(th)
        self.rf=c*d[0]+s*d[1]; self.rl=-s*d[0]+c*d[1]; self.rt=wrap(self.g('theta',self.H)-th)
        return self.rf,self.rl,self.rt
    def pan_y(self):
        self.rel(); return self.rf+0.045
    def carry_to(self,x,y=2.30):
        self.goto(y=y,th=-PI/2-self.rt,gap=0.08)
        self.goto(x=x,y=y,th=-PI/2-self.rt)
    def in_pan(self):
        """count objects sitting in the pan region"""
        x=self.g('x'); y=self.g('y'); self.rel()
        cy=y-self.rf
        n=0
        for o in self.smalls():
            ox,oy=self.g('x',o),self.g('y',o)
            if x-0.55<ox<x+0.05 and cy-0.02<oy<cy+0.55: n+=1
        return n

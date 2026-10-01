import numpy as np, math
from env_client import make_env
from scoop_lib import Scoop
PI=math.pi
def mk(env,seed):
    c=Scoop(env,seed); return c
def hp(c): return (round(c.g('x',c.H),4), round(c.g('y',c.H),4), round(c.g('theta',c.H),4), c.g('held',c.H))
def descend(c, dy=-0.03, lim=60):
    """step down until blocked; return final y"""
    for i in range(lim):
        y0=c.g('y')
        c.step([0,dy,0,0,0])
        if abs(c.g('y')-y0)<1e-4: return c.g('y'), True
    return c.g('y'), False
def close_g(c, n=20):
    for _ in range(n):
        c.step([0,0,0,0,-0.015])
        if c.g('held',c.H)>0.5: return True
    return c.g('held',c.H)>0.5

def route(c, x, y=1.0, th=None, arm=0.2, gap=0.08):
    import math
    th = -math.pi/2 if th is None else th
    c.goto(y=2.3, th=-math.pi/2, arm=arm, gap=gap, tol=0.01, maxit=60)
    c.goto(x=x, y=2.3, tol=0.01, maxit=90)
    c.goto(y=y, th=th, tol=0.01, maxit=90)
    return c.t

C=[(-0.035,0.033,0.41),(0.032,-0.002,0.424),(0.013,0.043,0.41),(-0.034,0.037,0.429),(0.022,0.022,0.43),(-0.009,0.03,0.415),(0.018,-0.01,0.441),(0.02,0.021,0.41),(-0.03,-0.022,0.409),(-0.026,-0.014,0.429),(0.041,0.017,0.41),(0.012,-0.033,0.429),(-0.008,-0.019,0.411),(-0.034,0.005,0.411),(-0.003,-0.007,0.43),(-0.019,0.062,0.41),(0.001,0.006,0.41),(0.015,-0.029,0.41),(-0.029,0.013,0.429),(0.035,-0.025,0.41)]
from pngdec import load
w,h,bpp,rows=load('mcp_renders/state_seed3.png')
def lab(r,g,b):
    if r>150 and g>150 and b<100:return 'Y'
    if r>60 and g<60 and b<60 and r>1.8*g:return 'R'
    if g>60 and r<60 and b<60:return 'G'
    if b>60 and r<60 and g<60:return 'B'
    return '?'
for i,(x,y,z) in enumerate(C,1):
    zt=z+0.01
    u=320-440*y; v=336+265*(-x)+410*(0.5-zt)
    ui,vi=round(u),round(v)
    s=''.join(lab(*rows[vi+dy][(ui+dx)*bpp:(ui+dx)*bpp+3]) for dy in (-1,0,1) for dx in (-1,0,1))
    print(i,'rgby'[(i-1)%4] if False else ['red','green','blue','yellow'][(i-1)%4], 'px',ui,vi,'crop',(ui-280)*8,(vi-335)*8, s)

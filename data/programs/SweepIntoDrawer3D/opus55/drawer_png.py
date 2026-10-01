import zlib, struct, numpy as np, sys
def read_png(fn):
    d=open(fn,'rb').read(); i=8; idat=b''
    while i<len(d):
        L,=struct.unpack('>I',d[i:i+4]); t=d[i+4:i+8]; c=d[i+8:i+8+L]; i+=12+L
        if t==b'IHDR': w,h,bd,ct=struct.unpack('>IIBB',c[:10])
        elif t==b'IDAT': idat+=c
    ch={2:3,6:4,0:1}[ct]; raw=zlib.decompress(idat); s=w*ch
    out=np.zeros((h,s),np.int32); prev=np.zeros(s,np.int32); p=0
    for y in range(h):
        f=raw[p]; line=np.frombuffer(raw[p+1:p+1+s],np.uint8).astype(np.int32); p+=1+s
        if f==0: cur=line
        elif f==2: cur=(line+prev)&255
        else:
            cur=np.zeros(s,np.int32)
            for x in range(s):
                a=cur[x-ch] if x>=ch else 0; b=prev[x]; cc=prev[x-ch] if x>=ch else 0
                if f==1: pr=a
                elif f==3: pr=(a+b)//2
                else:
                    pa=abs(b-cc); pb=abs(a-cc); pc=abs(a+b-2*cc)
                    pr=a if (pa<=pb and pa<=pc) else (b if pb<=pc else cc)
                cur[x]=(line[x]+pr)&255
        out[y]=cur; prev=cur
    return out.reshape(h,w,ch)[:,:,:3].astype(np.uint8)
def write_png(fn,a):
    h,w,_=a.shape; raw=b''.join(b'\x00'+a[y].tobytes() for y in range(h))
    def chunk(t,c): return struct.pack('>I',len(c))+t+c+struct.pack('>I',zlib.crc32(t+c)&0xffffffff)
    open(fn,'wb').write(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(a.tobytes() if False else raw))+chunk(b'IEND',b''))
if __name__=='__main__':
    fn,out,x0,y0,x1,y1,s=sys.argv[1],sys.argv[2],*map(int,sys.argv[3:8])
    a=read_png(fn)[y0:y1,x0:x1]; a=np.repeat(np.repeat(a,s,0),s,1); write_png(out,np.ascontiguousarray(a))

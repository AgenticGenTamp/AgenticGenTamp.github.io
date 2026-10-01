import zlib, struct, numpy as np, sys
def read_png(path):
    d=open(path,'rb').read(); assert d[:8]==b'\x89PNG\r\n\x1a\n'
    i=8; idat=b''; w=h=bd=ct=None
    while i<len(d):
        ln=struct.unpack('>I',d[i:i+4])[0]; typ=d[i+4:i+8]; data=d[i+8:i+8+ln]; i+=12+ln
        if typ==b'IHDR': w,h,bd,ct=struct.unpack('>IIBB',data[:10])
        elif typ==b'IDAT': idat+=data
        elif typ==b'IEND': break
    raw=zlib.decompress(idat)
    nc={0:1,2:3,3:1,4:2,6:4}[ct]; stride=w*nc
    out=np.zeros((h,stride),dtype=np.uint8); prev=np.zeros(stride,dtype=np.uint8); p=0
    for y in range(h):
        f=raw[p]; p+=1
        line=np.frombuffer(raw[p:p+stride],dtype=np.uint8).astype(np.int32).copy(); p+=stride
        if f==1:
            for x in range(nc,stride): line[x]=(line[x]+line[x-nc])&255
        elif f==2: line=(line+prev)&255
        elif f==3:
            for x in range(stride):
                a=line[x-nc] if x>=nc else 0
                line[x]=(line[x]+((a+prev[x])>>1))&255
        elif f==4:
            for x in range(stride):
                a=int(line[x-nc]) if x>=nc else 0
                b=int(prev[x]); c=int(prev[x-nc]) if x>=nc else 0
                pp=a+b-c; pa,pb,pc=abs(pp-a),abs(pp-b),abs(pp-c)
                pr=a if (pa<=pb and pa<=pc) else (b if pb<=pc else c)
                line[x]=(line[x]+pr)&255
        out[y]=line.astype(np.uint8); prev=out[y]
    return out.reshape(h,w,nc)
def write_png(path,arr):
    h,w,nc=arr.shape; ct={1:0,3:2,4:6}[nc]
    raw=b''.join(b'\x00'+arr[y].tobytes() for y in range(h))
    def chunk(t,d): 
        c=struct.pack('>I',len(d))+t+d; return c+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    open(path,'wb').write(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,ct,0,0,0))+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))
def crop_zoom(src,dst,x0,y0,x1,y1,z=4):
    a=read_png(src)[y0:y1,x0:x1,:3]
    a=np.repeat(np.repeat(a,z,axis=0),z,axis=1)
    write_png(dst,a); print(dst,a.shape)
if __name__=='__main__':
    a=sys.argv
    if a[1]=='crop': crop_zoom(a[2],a[3],*[int(v) for v in a[4:9]])
    else: print(read_png(a[2]).shape)

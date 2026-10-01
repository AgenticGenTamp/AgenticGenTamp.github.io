import zlib, struct
import numpy as np

def read_png(path):
    data = open(path,'rb').read()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    pos = 8; idat = b''; w=h=bd=ct=None; plte=None
    while pos < len(data):
        ln = struct.unpack('>I', data[pos:pos+4])[0]; typ = data[pos+4:pos+8]
        chunk = data[pos+8:pos+8+ln]; pos += 12+ln
        if typ==b'IHDR':
            w,h,bd,ct,_,_,_ = struct.unpack('>IIBBBBB', chunk)
        elif typ==b'IDAT': idat += chunk
        elif typ==b'PLTE': plte = np.frombuffer(chunk,dtype=np.uint8).reshape(-1,3)
        elif typ==b'IEND': break
    raw = zlib.decompress(idat)
    nch = {0:1,2:3,3:1,4:2,6:4}[ct]
    assert bd==8
    stride = w*nch
    out = np.zeros((h,stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    i=0
    for y in range(h):
        f = raw[i]; i+=1
        line = np.frombuffer(raw[i:i+stride], dtype=np.uint8).astype(np.int32).copy(); i+=stride
        if f==0: pass
        elif f==1:
            for x in range(nch, stride): line[x] = (line[x]+line[x-nch])&255
        elif f==2: line = (line+prev.astype(np.int32))&255
        elif f==3:
            for x in range(stride):
                a = line[x-nch] if x>=nch else 0
                line[x] = (line[x]+((a+int(prev[x]))>>1))&255
        elif f==4:
            for x in range(stride):
                a = int(line[x-nch]) if x>=nch else 0
                b = int(prev[x]); c = int(prev[x-nch]) if x>=nch else 0
                p = a+b-c; pa=abs(p-a); pb=abs(p-b); pc=abs(p-c)
                pr = a if (pa<=pb and pa<=pc) else (b if pb<=pc else c)
                line[x] = (line[x]+pr)&255
        prev = line.astype(np.uint8)
        out[y] = prev
    img = out.reshape(h,w,nch)
    if ct==3: img = plte[img[:,:,0]]
    return img[:,:,:3]

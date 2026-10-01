import zlib, struct, numpy as np
def write_png(fn, img):
    h, w, _ = img.shape
    raw = b''.join(b'\x00' + img[i].astype(np.uint8).tobytes() for i in range(h))
    def chunk(t, d): 
        c = struct.pack('>I', len(d)) + t + d
        return c + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
    open(fn, 'wb').write(png)
def render(ap, fn, box=(0, 2.5, 0, 2.5), res=400, extra_polys=()):
    x0, x1, y0, y1 = box
    xs = np.linspace(x0, x1, res); ys = np.linspace(y1, y0, res)
    X, Y = np.meshgrid(xs, ys)
    img = np.full((res, res, 3), 255.0)
    def fill(C, col):
        inside = np.ones_like(X, bool)
        for k in range(4):
            a = C[k]; b = C[(k+1)%4]
            cr = (b[0]-a[0])*(Y-a[1]) - (b[1]-a[1])*(X-a[0])
            inside &= cr >= 0
        inside2 = np.ones_like(X, bool)
        for k in range(4):
            a = C[k]; b = C[(k+1)%4]
            cr = (b[0]-a[0])*(Y-a[1]) - (b[1]-a[1])*(X-a[0])
            inside2 &= cr <= 0
        img[inside | inside2] = col
    for n in ap.rects:
        if n == 'target_region': fill(ap.corners(n), (180, 255, 180))
    for n in ap.rects:
        if n == 'target_region': continue
        fill(ap.corners(n), (255, 60, 60) if n == 'target_block' else (120, 120, 120))
    q = ap.q
    m = (X-q[0])**2 + (Y-q[1])**2 < 0.1**2
    img[m] = (60, 60, 255)
    for P, col in extra_polys: fill(P, col)
    write_png(fn, img)

"""Minimal pure-Python PNG decoder.

Supports 8-bit greyscale / greyscale+alpha / RGB / RGBA / palette PNGs with
the five standard scanline filters.  Only depends on ``zlib``, ``struct`` and
``numpy``.

Main entry point::

    from pngread import read_png
    img = read_png("pic.png")      # -> uint8 array, shape (H, W, 3)
    rgba = read_png("pic.png", rgba=True)   # -> shape (H, W, 4)
"""

import struct
import zlib

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# color type -> number of samples per pixel
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _iter_chunks(data):
    """Yield (chunk_type, chunk_data) tuples from a PNG byte string."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG file (bad signature)")
    pos = len(PNG_SIGNATURE)
    end = len(data)
    while pos + 8 <= end:
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        start = pos + 8
        stop = start + length
        if stop + 4 > end:
            raise ValueError("truncated PNG chunk %r" % (ctype,))
        yield ctype, data[start:stop]
        pos = stop + 4  # skip the CRC
        if ctype == b"IEND":
            break


def _paeth(a, b, c):
    """Vectorised Paeth predictor (inputs are int arrays)."""
    p = a + b - c
    pa = np.abs(p - a)
    pb = np.abs(p - b)
    pc = np.abs(p - c)
    out = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
    return out


def _unfilter(raw, height, width, bpp):
    """Undo PNG scanline filtering.

    ``raw`` is the decompressed stream, one filter byte per scanline followed
    by ``stride`` data bytes.  Returns a (height, stride) uint8 array.
    """
    stride = width * bpp
    expected = height * (stride + 1)
    if len(raw) < expected:
        raise ValueError("decompressed data too short: %d < %d"
                         % (len(raw), expected))
    buf = np.frombuffer(raw[:expected], dtype=np.uint8)
    buf = buf.reshape(height, stride + 1)
    filters = buf[:, 0]
    out = np.zeros((height, stride), dtype=np.int32)
    prev = np.zeros(stride, dtype=np.int32)

    for y in range(height):
        ft = int(filters[y])
        line = buf[y, 1:].astype(np.int32)
        if ft == 0:                                  # None
            cur = line
        elif ft == 2:                                # Up
            cur = (line + prev) & 0xFF
        elif ft == 1:                                # Sub
            cur = line
            for x in range(bpp, stride, bpp):
                cur[x:x + bpp] = (cur[x:x + bpp] + cur[x - bpp:x]) & 0xFF
        elif ft == 3:                                # Average
            cur = line
            cur[:bpp] = (cur[:bpp] + (prev[:bpp] >> 1)) & 0xFF
            for x in range(bpp, stride, bpp):
                cur[x:x + bpp] = (cur[x:x + bpp]
                                  + ((cur[x - bpp:x] + prev[x:x + bpp]) >> 1)) & 0xFF
        elif ft == 4:                                # Paeth
            cur = line
            cur[:bpp] = (cur[:bpp] + prev[:bpp]) & 0xFF
            for x in range(bpp, stride, bpp):
                cur[x:x + bpp] = (cur[x:x + bpp]
                                  + _paeth(cur[x - bpp:x],
                                           prev[x:x + bpp],
                                           prev[x - bpp:x])) & 0xFF
        else:
            raise ValueError("unknown filter type %d on row %d" % (ft, y))
        out[y] = cur
        prev = cur

    return out.astype(np.uint8)


def read_png(path, rgba=False):
    """Decode an 8-bit PNG file into a numpy uint8 array.

    Returns shape (H, W, 3) by default, or (H, W, 4) when ``rgba`` is True.
    """
    with open(path, "rb") as fh:
        data = fh.read()

    idat = []
    palette = None
    trns = None
    width = height = None
    depth = color_type = None

    for ctype, cdata in _iter_chunks(data):
        if ctype == b"IHDR":
            (width, height, depth, color_type, comp,
             filt, interlace) = struct.unpack(">IIBBBBB", cdata[:13])
            if depth != 8:
                raise ValueError("only 8-bit PNGs supported (got %d)" % depth)
            if comp != 0 or filt != 0:
                raise ValueError("unsupported compression/filter method")
            if interlace != 0:
                raise ValueError("interlaced PNGs are not supported")
            if color_type not in _CHANNELS:
                raise ValueError("bad color type %d" % color_type)
        elif ctype == b"PLTE":
            palette = np.frombuffer(cdata, dtype=np.uint8).reshape(-1, 3)
        elif ctype == b"tRNS":
            trns = np.frombuffer(cdata, dtype=np.uint8)
        elif ctype == b"IDAT":
            idat.append(cdata)
        elif ctype == b"IEND":
            break

    if width is None:
        raise ValueError("no IHDR chunk found")
    if not idat:
        raise ValueError("no IDAT chunks found")

    raw = zlib.decompress(b"".join(idat))
    nch = _CHANNELS[color_type]
    planes = _unfilter(raw, height, width, nch).reshape(height, width, nch)

    if color_type == 3:                              # palette
        if palette is None:
            raise ValueError("palette PNG without PLTE chunk")
        idx = planes[:, :, 0]
        out = palette[idx]
        if trns is not None:
            alpha = np.full(idx.shape, 255, dtype=np.uint8)
            n = min(len(trns), len(palette))
            lut = np.full(len(palette), 255, dtype=np.uint8)
            lut[:n] = trns[:n]
            alpha = lut[idx]
            out = np.dstack([out, alpha])
    elif color_type == 0:                            # grey
        out = np.repeat(planes, 3, axis=2)
    elif color_type == 4:                            # grey + alpha
        out = np.dstack([np.repeat(planes[:, :, :1], 3, axis=2),
                         planes[:, :, 1:2]])
    else:                                            # 2 (RGB) or 6 (RGBA)
        out = planes

    if rgba:
        if out.shape[2] == 3:
            alpha = np.full(out.shape[:2] + (1,), 255, dtype=np.uint8)
            out = np.concatenate([out, alpha], axis=2)
        return np.ascontiguousarray(out)
    return np.ascontiguousarray(out[:, :, :3])


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        a = read_png(p)
        print(p, a.shape, a.dtype)

"""Force single-threaded OpenBLAS at runtime (tiny matrices; threading only hurts)."""
import ctypes


def limit_blas_threads(n=1):
    done = False
    try:
        import numpy  # noqa: F401  (ensure loaded)
        with open("/proc/self/maps") as f:
            paths = {line.split()[-1] for line in f if "openblas" in line.lower()}
        for pth in paths:
            try:
                lib = ctypes.CDLL(pth)
            except OSError:
                continue
            for fn in ("openblas_set_num_threads", "openblas_set_num_threads64_",
                       "scipy_openblas_set_num_threads", "scipy_openblas_set_num_threads64_"):
                f = getattr(lib, fn, None)
                if f is not None:
                    f(int(n))
                    done = True
    except Exception:
        pass
    return done

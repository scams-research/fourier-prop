import timeit
import numpy as np
import scipp as sc

from fourierror import numerical

import paths

M = 10

rng = np.random.default_rng(42)

N_BOTH = 2 ** np.arange(start=7, stop=13, step=1, dtype=int)
N_FFT = 2 ** np.arange(start=13, stop=17, step=1, dtype=int)
N_SETUP = 2 ** np.arange(start=7, stop=12, step=1, dtype=int)

def timed(fn, target=0.02, repeats=7):
    """Minimum per-call time, with `number` chosen automatically.
 
    A single FFT call can be a few microseconds, close to the clock's
    resolution, so batching is necessary; but the batch size must then
    be divided out. timeit.Timer.autorange picks a `number` that takes
    at least `target` seconds, and the result is divided by it.
    """
    timer = timeit.Timer(fn)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=repeats, number=number)) / number

times = np.zeros((N_BOTH.size, 2))

for i, n in enumerate(N_BOTH):
    n = int(n)
    samples = rng.standard_normal((n, M))
    dft_matrix = np.fft.fft(np.eye(n))

    times[i, 0] = timed(lambda: (2 / n) * (dft_matrix @ samples))
    times[i, 1] = timed(lambda: (2 / n) * np.fft.fft(samples, axis=0))

    print(
        f"N={n:6d}  dft {times[i, 0] * 1e6:10.1f} us   "
        f"fft {times[i, 1] * 1e6:9.1f} us   "
        f"ratio {times[i, 0] / times[i, 1]:7.1f}"
    )

    del dft_matrix

times_fft = np.zeros(N_FFT.size)
 
for i, n in enumerate(N_FFT):
    n = int(n)
    samples = rng.standard_normal((n, M))
    times_fft[i] = timed(lambda: (2 / n) * np.fft.fft(samples, axis=0))
    print(f"N={n:6d}  fft only {times_fft[i] * 1e6:9.1f} us")

n_all = np.concatenate([N_BOTH, N_FFT])
t_all = np.concatenate([times[:, 1], times_fft])
print(t_all)

np.savez(paths.data / "numerical_N.npz", N=n_all, time_dft=times[:, 0], time_fft=t_all)

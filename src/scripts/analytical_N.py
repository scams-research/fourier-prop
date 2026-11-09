import timeit
import numpy as np
import scipp as sc

import paths

rng = np.random.default_rng(42)

repeats = int(2**6)

def to_run(n, data):
    cov = np.diag(data.variances)
        
    theta = (
        -2
        * np.pi
        * (
            np.arange(data.values.size)[:, np.newaxis]
            * np.arange(data.values.size)[:, np.newaxis].T
        )
        / data.values.size
    )

    cos = (2 / data.values.size) * np.cos(theta)
    sin = (2 / data.values.size) * np.sin(theta)

    dft_matrix = cos + 1j * sin

    f = dft_matrix @ data.values

    var_real = cos @ cov @ cos.T
    var_imag = sin @ cov @ sin.T

def to_run_fft(n, data):
    cov = np.diag(data.variances)
        
    f = np.fft.fft(data.values) * (2 / n)

    p = np.fft.fft2(cov)
    q = np.fft.fft(np.fft.ifft(cov, axis=1) * n, axis=0)

    scale = (2 / n) ** 2

    var_real = scale * (p.real + q.real) / 2
    var_imag = scale * (q.real - p.real) / 2

N = 2 ** np.arange(start=6, stop=13, step=1, dtype=int)

def n_repeats(n):
    if n <= 512:
        return 20
    if n <= 2048:
        return 5
    return 3

times = np.zeros((N.size, 2))

for i, n in enumerate(N):
    X = sc.linspace(dim='x', start=0, stop=2*np.pi, num=n, unit='rad')
    Y = sc.sin(2 * X) + 2 * sc.cos(10 * X) + 0.6 * sc.sin(20 * X)
    Y += sc.array(dims=['x'], values=rng.standard_normal(n) * 0.1 * Y.values)
    Y.variances = sc.abs(Y * 0.1)
    Y.unit = sc.Unit('s') 
    data = sc.DataArray(data=Y, coords={'x': X})

    r = n_repeats(n)
        
    dft_timer = timeit.Timer(
        lambda: to_run(n, data)
    ).repeat(repeat=r, number=1)

    fft_timer = timeit.Timer(
        lambda: to_run_fft(n, data)
    ).repeat(repeat=r, number=1)

    times[i, 0] = np.min(dft_timer)
    times[i, 1] = np.min(fft_timer)
    print(times[i])

np.savez(paths.data / "analytical_N.npz", N=N, times=times)

lg = np.log2(N)
tail = N >= 2 ** 10
for j, lab in enumerate(["dft", "fft"]):
    full = np.polyfit(lg, np.log2(times[:, j]), 1)[0]
    asym = np.polyfit(lg[tail], np.log2(times[tail, j]), 1)[0]
    print(f"{lab}: slope over full range {full:.2f}, over N >= 2^10 {asym:.2f}")
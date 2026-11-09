import numpy as np
import matplotlib.pyplot as plt

import _fig_params as fp
import paths

num_M = np.load(paths.data / "numerical_M.npz")
num_N = np.load(paths.data / "numerical_N.npz")
ana_N = np.load(paths.data / "analytical_N.npz")

DFT_KW = dict(color=fp.colors[0], marker='o')
FFT_KW = dict(color=fp.colors[1], marker='o')
GUIDE_KW = dict(ls='--', color='k', alpha=0.5, marker='', lw=0.5)


def guide(ax, x, y, f, label, below=False, factor=3):
    """Reference line through the last data point, offset by `factor`.

    The line spans the larger-x half of the data, where the asymptotic
    scaling applies, and is labelled at its right-hand end.
    """
    x = np.asarray(x, dtype=float)
    xs = x[x.size // 2:]
    shift = 1 / factor if below else factor
    ys = y[-1] * shift * f(xs) / f(xs[-1])
    ax.plot(xs, ys, **GUIDE_KW)
    LABELS.setdefault(ax, []).append((xs[-1], ys[-1], label, 'k'))


def end_label(ax, x, y, label, color):
    """Label a line directly at its right-hand end."""
    LABELS.setdefault(ax, []).append((x[-1], y[-1], label, color))


def place_labels(ax, gap=0.09):
    """Draw the queued labels, spreading them vertically so none overlap.

    Positions are handled in axes-fraction units on the (log) y axis.
    """
    lo, hi = np.log10(ax.get_ylim())
    items = sorted(LABELS.get(ax, []), key=lambda i: i[1])
    pos = [(np.log10(y) - lo) / (hi - lo) for _, y, _, _ in items]
    for _ in range(50):
        for j in range(1, len(pos)):
            overlap = gap - (pos[j] - pos[j - 1])
            if overlap > 0:
                pos[j - 1] -= overlap / 2
                pos[j] += overlap / 2
    for (x, _, label, color), f in zip(items, pos):
        ax.text(x * 1.15, 10 ** (lo + f * (hi - lo)), label, color=color, ha='left', va='center')


LABELS = {}


fig = plt.figure(figsize=(fp.PAGE_WIDTH, 2))
gs = plt.GridSpec(1, 3, figure=fig, wspace=0.45)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

# (a) numerical, varying M
ax = axes[0]
M, t = num_M["M"], num_M["times"]
ax.plot(M, t[:, 0], **DFT_KW)
ax.plot(M, t[:, 1], **FFT_KW)
end_label(ax, M, t[:, 0], 'DFT', fp.colors[0])
end_label(ax, M, t[:, 1], 'FFT', fp.colors[1])
guide(ax, M, t[:, 1], lambda m: m, r'$\mathcal{O}(M)$', below=True)
ax.set_xlabel('$M$')

# (b) numerical, varying N
ax = axes[1]
N, t_dft, t_fft = num_N["N"], num_N["time_dft"], num_N["time_fft"]
N_dft = N[:t_dft.size]
ax.plot(N_dft, t_dft, **DFT_KW)
ax.plot(N, t_fft, **FFT_KW)
end_label(ax, N_dft, t_dft, 'DFT', fp.colors[0])
end_label(ax, N, t_fft, 'FFT', fp.colors[1])
guide(ax, N_dft, t_dft, lambda n: n ** 2, r'$\mathcal{O}(N^2)$')
guide(ax, N, t_fft, lambda n: n * np.log(n), r'$\mathcal{O}(N\,\log N)$', below=True)
ax.set_xlabel('$N$')

# (c) analytical, varying N
ax = axes[2]
N, t = ana_N["N"], ana_N["times"]
ax.plot(N, t[:, 0], **DFT_KW)
ax.plot(N, t[:, 1], **FFT_KW)
end_label(ax, N, t[:, 0], 'DFT', fp.colors[0])
end_label(ax, N, t[:, 1], 'FFT', fp.colors[1])
guide(ax, N, t[:, 0], lambda n: n ** 3, r'$\mathcal{O}(N^3)$')
guide(ax, N, t[:, 1], lambda n: n ** 2 * np.log(n), r'$\mathcal{O}(N^2 \log N)$', below=True)
ax.set_xlabel('$N$')

for i, ax in enumerate(axes):
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_ylabel('Time / s')
    fp.panel_label(ax, i)
    place_labels(ax)

plt.savefig(paths.figures / "scaling.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close()

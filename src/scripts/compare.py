import numpy as np
import matplotlib.pyplot as plt
import scipp as sc

from fourierror import numerical, analytical

import _fig_params as fp
import paths

X = sc.linspace(dim='x', start=0, stop=2*np.pi, num=50, unit='rad', endpoint=False)
Y = sc.sin(2 * X) + 2 * sc.cos(10 * X) + 0.6 * sc.sin(20 * X)
Y.variances = (0.3 * Y.values) ** 2
Y.unit = sc.Unit('s') 
data = sc.DataArray(data=Y, coords={'x': X})

F_num = numerical.dft(data, 'x', 1_000_000, random_state=42)
F_ana = analytical.dft(data, 'x')

fig = plt.figure(figsize=(fp.PAGE_WIDTH*0.8, 2.5))
gs = plt.GridSpec(3, 2, figure=fig, wspace=0.25, hspace=0.6, width_ratios=[1.8, 1], left=0.07, right=0.98)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0]) 
ax3 = fig.add_subplot(gs[2, 0])

ax1.errorbar(data.coords['x'].values, data.values, (data.variances ** 0.5) * 1.96, color=fp.colors[0], ls='', marker='.', elinewidth=0.8, label=r'$x(\phi)$')

ax2.errorbar(F_num['real'].coords['omega'].values, F_num['real'].values, (F_num['real'].variances ** 0.5) * 1.96, color=fp.colors[1], ls='', marker='.', elinewidth=0.8, label=r'$\operatorname{Re}[\mathcal{F}(\omega)]$')
ax2.errorbar(F_num['imag'].coords['omega'].values, F_num['imag'].values, (F_num['imag'].variances ** 0.5) * 1.96, color=fp.colors[2], ls='', marker='.', elinewidth=0.8, label=r'$\operatorname{Im}[\mathcal{F}(\omega)]$')

ax3.errorbar(F_ana['real'].coords['omega'].values, F_ana['real'].values, (F_ana['real'].variances ** 0.5) * 1.96, color=fp.colors[1], ls='', marker='.', elinewidth=0.8)
ax3.errorbar(F_ana['imag'].coords['omega'].values, F_ana['imag'].values, (F_ana['imag'].variances ** 0.5) * 1.96, color=fp.colors[2], ls='', marker='.', elinewidth=0.8)

ax1.set_xlabel(r'$\phi$')
ax2.set_xlabel(r'$\omega$')
ax3.set_xlabel(r'$\omega$')

ax1.set_ylabel(r'$x(\phi)$')
ax2.set_ylabel(r'$\mathcal{F}(\omega)_{\mathrm{num}}$')
ax3.set_ylabel(r'$\mathcal{F}(\omega)_{\mathrm{ana}}$')

ax4 = fig.add_subplot(gs[:, 1])

ax4.set_aspect('equal')
ax4.scatter(F_num['real'].variances * 1000, F_ana['real'].variances * 1000, color=fp.colors[1], alpha=0.5)
ax4.scatter(F_num['imag'].variances * 1000, F_ana['imag'].variances * 1000, color=fp.colors[2], alpha=0.5)
lim_max = 1.05 * 1000 * max(
    F_num['real'].variances.max(), F_ana['real'].variances.max(),
    F_num['imag'].variances.max(), F_ana['imag'].variances.max(),
)
lims = [-0.05 * lim_max, lim_max]
ax4.plot(lims, lims, color='k', ls='--', alpha=0.5, lw=0.5)

ax4.set_xlim(*lims)
ax4.set_ylim(*lims)

ax4.set_xlabel(r'$\sigma^2[\mathcal{F}(\omega)_{\mathrm{num}}]$ / $10^{-3}$')
ax4.set_ylabel(r'$\sigma^2[\mathcal{F}(\omega)_{\mathrm{ana}}]$ / $10^{-3}$')

fig.legend(bbox_to_anchor=[0.47, 1], ncols=3)

for i, ax in enumerate([ax1, ax2, ax3, ax4]):
    fp.panel_label(ax, i)

plt.savefig(paths.figures / "compare.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close()

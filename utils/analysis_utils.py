import numpy as np
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def quantile(x, q, w=None, is_sorted=False, estimate_underlying_quantile=False):
    n = len(x)
    x = np.array(x)
    q = np.array(q)

    # If we estimate_underlying_quantile, we refer to min(x),max(x) not as
    #  quantiles 0,1, but rather as quantiles 1/(n+1),n/(n+1) of the
    #  underlying distribution from which x is sampled.
    if estimate_underlying_quantile and n > 1:
        q = q * (n+1)/(n-1) - 1/(n-1)
        q = np.clip(q, 0, 1)

    # Unweighted quantiles
    if w is None:
        return np.percentile(x, 100*q)

    # Weighted quantiles
    x = np.array(x)
    w = np.array(w)
    if not is_sorted:
        ids = np.argsort(x)
        x = x[ids]
        w = w[ids]
    w = np.cumsum(w) - 0.5*w
    w -= w[0]
    w /= w[-1]
    return np.interp(q, w, x)

def plot_quantiles(x, ax=None, q=None, showmeans=True, means_args=None,
                   dots=False, **kwargs):
    if ax is None: ax = Axes(1,1)[0]
    if q is None: q = np.arange(101) / 100
    m = np.mean(x)
    x = quantile(x, q)
    h = ax.plot(100*q, x, '.-' if dots else '-', **kwargs)
    if showmeans:
        if means_args is None: means_args = {}
        ax.axhline(m, linestyle='--', color=h[0].get_color(), **means_args)
    return ax

def qplot(data, y, x=None, hue=None, ax=None, **kwargs):
    if ax is None: ax = Axes(1,1)[0]

    if hue is None:
        plot_quantiles(data[y], ax=ax, **kwargs)
        same_samp_size = True
        n = len(data)
    else:
        hue_vals = pd.unique(data[hue].values)
        same_samp_size = len(pd.unique([(data[hue]==hv).sum()
                                        for hv in hue_vals])) == 1
        n = int(len(data) // len(hue_vals))
        for hv in hue_vals:
            d = data[data[hue]==hv]
            lab = hv
            if not same_samp_size:
                lab = f'{lab} (n={len(d):d})'
            plot_quantiles(d[y], ax=ax, label=lab, **kwargs)
        ax.legend(fontsize=13)

    xlab = 'quantile [%]'
    if x: xlab = f'{x} {xlab}'
    if same_samp_size: xlab = f'{xlab}\n({n:d} samples)'
    labels(ax, xlab, y, fontsize=15)

    return ax

def smooth(y, n=10, deg=2):
    n = min(n, len(y))
    if n%2 == 0: n -= 1
    return signal.savgol_filter(y, n, deg)

def labels(ax, xlab=None, ylab=None, title=None, fontsize=12):
    if isinstance(fontsize, int):
        fontsize = 3*[fontsize]
    if xlab is not None:
        ax.set_xlabel(xlab, fontsize=fontsize[0])
    if ylab is not None:
        ax.set_ylabel(ylab, fontsize=fontsize[1])
    if title is not None:
        ax.set_title(title, fontsize=fontsize[2])

class Axes:
    def __init__(self, N, W=2, axsize=(5,3.5), grid=1, fontsize=13):
        self.fontsize = fontsize
        self.N = N
        self.W = W
        self.H = int(np.ceil(N/W))
        self.axs = plt.subplots(self.H, self.W,
                                figsize=(self.W*axsize[0], self.H*axsize[1]))[1]
        for i in range(self.N):
            if grid == 1:
                self[i].grid(color='k', linestyle=':', linewidth=0.3)
            elif grid ==2:
                self[i].grid()
        for i in range(self.N, self.W*self.H):
            self[i].axis('off')

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        if self.H == 1 and self.W == 1:
            return self.axs
        elif self.H == 1 or self.W == 1:
            return self.axs[item]
        return self.axs[item//self.W, item % self.W]

    def labs(self, item, *args, **kwargs):
        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = self.fontsize
        labels(self[item], *args, **kwargs)

def show_trajectories(res, n_trajectories=5, fontsize=15, cmap='Blues', axargs=None):
    # set axes
    default_axargs = dict(W=3, axsize=(6,4))
    if axargs is not None:
        for k,v in axargs.items():
            default_axargs[k] = v
    axs = Axes(n_trajectories, **default_axargs)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize())

    # extract returns
    rets = sorted(res.ret.values)
    ids = np.linspace(0, len(res)-1, n_trajectories).astype(int)
    rets = np.take(rets, ids)

    # choose trajectories and visualize
    for i, ret in enumerate(rets):
        r = res[res.ret == ret]
        episode = r.episode.values[0]
        r = r[r.episode == episode]
        show_trajectory(r, axs[i], fontsize=fontsize, cmap=cmap, sm=sm)

    plt.tight_layout()
    return axs

def show_trajectory(res, ax=None, fontsize=15, cmap='Blues', sm=None):
    if ax is None:
        ax = Axes(1, 1)
    if sm is None:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize())

    x, y, k = res.x.values, res.y.values, res.k.values
    counts = np.arange(len(x) - 1)
    sm.set_array(counts)

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
    ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], counts, scale_units='xy', angles='xy', scale=1,
              color='k', cmap=cmap, clim=(-5, counts[-1]), headwidth=6, headlength=8)
    plt.colorbar(sm, ax=ax)
    ax.plot(x[-1:], y[-1:], 'rs', markersize=14)
    ax.plot(x[:1], y[:1], 'g>', markersize=16)
    ax.plot(x, y, 'yo', markersize=7)
    for i, txt in enumerate(range(len(x))):
        ax.annotate(f'{i:d}:{k[i]:d}', (x[i], y[i]), fontsize=fontsize - 3)
    labels(ax, 'x', 'y', f'iter={res.iteration.values[0]}, env={res.i_env.values[0]}, return={res.ret.values[0]:.2f}',
             fontsize=fontsize)

    return ax

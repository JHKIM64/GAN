import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.colors import ListedColormap, TwoSlopeNorm
c_grade = [15,25,75]

def imaging(title, images, text, C_max) :
    plot(title[0], images[0], images[0], False, text, C_max)
    plot(title[1], images[1], images[1], False, text, C_max)
    plot(title[2], images[2], images[1], False, text, C_max)
    plot('Residual',images[2], images[1], True, text, C_max)

def plot(title, img, target, compare, text,C_max) :
    n = 49
    s = 25
    e = 135.5
    w = 100

    lon = np.linspace(w, e, int((e - w) * 2) + 1)
    lat = np.linspace(s, n, (n - s) * 2)

    X, Y = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(18, 12))
    box = [100, 135.5, 25, 49]
    scale = '50m'
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(box, crs=ccrs.PlateCarree())
    ax.coastlines(scale)
    ax.set_xticks(np.arange(box[0], box[1], 0.5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(box[2], box[3], 0.5), crs=ccrs.PlateCarree())
    ax.grid(b=True)
    plt.title(title + text, fontdict={'size': 13})

    C = img[:, :, 0]

    if compare :
        U = img[:, :, 1]
        V = img[:, :, 2]
        Q = ax.quiver(X, Y, U, V, scale=1, scale_units='inches')
        C = (img[:, :, 0] - target[:, :, 0]) * C_max

        c = ax.pcolormesh(X, Y, C, cmap=cm.get_cmap('bwr',20))
        fig.colorbar(c)
    else :
        U = img[:, :, 1]
        V = img[:, :, 2]
        Q = ax.quiver(X, Y, U, V, scale=1, scale_units='inches')
        C = (img[:, :, 0] - target[:, :, 0]) * C_max
        c = ax.pcolormesh(X, Y, C,)
        fig.colorbar(c)

    plt.savefig('/home/intern01/jhk/interpolation/figure/' + title + text +'.png')
"""
iSPEX 2 plotting functions
"""

import numpy as np
from matplotlib import pyplot as plt
from spectacle import plot as spectacle_plot
from .wavelength import fluorescent_lines
from matplotlib import pyplot as plt, patheffects as pe, ticker

offsets = [650, 1550]

def show_RGBG(data, colour=None, colorbar_label="", saveto=None, **kwargs):
    fig, axs = plt.subplots(nrows=4, sharex=True, sharey=True, figsize=(5,5), squeeze=True, gridspec_kw={"wspace":1, "hspace":0})
    for ax, data_c, c in zip(axs.ravel(), data, spectacle_plot.RGBG2):
        img = ax.imshow(data_c, cmap=spectacle_plot.cmaps[c+"r"], **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
    spectacle_plot._saveshow(saveto, bbox_inches="tight")


def plot_fluorescent_lines(y, lines, lines_fit, saveto=None):
    plt.figure(figsize=(7, 4))

    p_eff = [pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]
    for j, c in enumerate("rgb"):
        plt.scatter(lines[j], y, s=25, c=c, alpha=0.8)
        plt.plot(lines_fit[j], y, c=c, path_effects=p_eff)

    plt.title("Locations of RGB maxima")
    plt.xlabel("Line center") # x
    plt.ylabel("Row along spectrum") # y
    plt.axis("tight")
    plt.grid(ls="--")
    spectacle_plot._saveshow(saveto, bbox_inches="tight")


def plot_fluorescent_lines_double(y2, lines2, lines_fit2, saveto=None):
    p_eff = [pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]

    plt.figure(figsize=(10, 3))
    for offset, y, lines, lines_fit in zip(offsets, y2, lines2, lines_fit2):
        for j, c in enumerate("rgb"):
            plt.scatter(lines[j], y+offset, s=25, c=c, alpha=0.8)
            plt.plot(lines_fit[j], y+offset, c=c, path_effects=p_eff)

    plt.title("Locations of RGB maxima")
    plt.xlabel("Line center [px]") # x
    plt.ylabel("Row along spectrum [px]") # y
    plt.gca().invert_yaxis()
    plt.grid(ls="--")
    spectacle_plot._saveshow(saveto, bbox_inches="tight")


def plot_fluorescent_lines_dispersion(y2, lines2, lines_fit2, dispersion2, saveto=None):
    p_eff = [pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]

    fig, axs = plt.subplots(ncols=2, figsize=(10, 3), gridspec_kw={"width_ratios": (4,1), "hspace": 0, "wspace": 0.05}, sharey=True)
    for offset, y, lines, lines_fit, dispersion in zip(offsets, y2, lines2, lines_fit2, dispersion2):
        for j, c in enumerate("rgb"):
            axs[0].scatter(lines[j], y+offset, s=25, c=c, alpha=0.8)
            axs[0].plot(lines_fit[j], y+offset, c=c, path_effects=p_eff)

        axs[1].plot(dispersion, y+offset, c='k', lw=5)

    axs[0].set_title("Locations of RGB maxima")
    axs[0].set_xlabel("Line center [px]") # x
    axs[0].set_ylabel("Row along spectrum [px]") # y
    axs[0].invert_yaxis()
    axs[1].tick_params(axis="y", left=False)
    axs[1].set_xlabel("Dispersion [nm/px]")
    for ax in axs:
        ax.grid(ls="--")

    spectacle_plot._saveshow(saveto, bbox_inches="tight")

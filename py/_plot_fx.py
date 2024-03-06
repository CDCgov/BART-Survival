import matplotlib.pyplot as plt
import numpy as np


def plots1(
    meta,
    sv_true,
    k_sv,
    pb_sv,
    r_sv,
    figsize=(8,8)
):

    fig, ax = plt.subplots(1,1,figsize=figsize)

    ttt = meta[1]
    t = meta[2]
    ax.step(ttt, sv_true, color="black", label="t0", where="mid")
    # kp
    ax.step(t, k_sv[0], color = "coral", label = "k0", alpha=0.8, where="mid")
    ax.step(t, k_sv[1][0], color = "coral", alpha=0.5, linestyle="dashed", where="mid")
    ax.step(t, k_sv[1][1], color = "coral", alpha=0.5, linestyle="dashed", where="mid")
    # pb
    ax.step(t, pb_sv[0][0], label="pb0", color = "blue", alpha=0.5, where="mid")
    ax.step(t, pb_sv[0][1][0], color = "blue", alpha=0.5, linestyle="dashed", where="mid")
    ax.step(t, pb_sv[0][1][1], color = "blue", alpha=0.5, linestyle="dashed", where="mid")
    # r
    ax.step(t, r_sv[0], color="green", label="r0", alpha=0.5, where="mid")
    ax.step(t, r_sv[1][0], color="green",  alpha=0.5, linestyle="dashed", where="mid")
    ax.step(t, r_sv[1][1], color="green",  alpha=0.5, linestyle="dashed", where="mid")
    ax.legend()
    return fig


def plots2(
    meta,
    sv_true,
    k_sv,
    pb_sv,
    r_sv,
    figsize=(16,8)
):

    fig, ax = plt.subplots(1,2,figsize=figsize)

    ttt = meta[1]
    t = meta[2]
    ax[0].step(ttt, sv_true[0], color="black", label="t0", where="mid")
    ax[1].step(ttt, sv_true[1], color="black", label="t1", where="mid")
    # kp
    ax[0].step(t, k_sv[0][0], color = "coral", label = "k0", alpha=0.8, where="mid")
    ax[0].step(t, k_sv[0][1][0], color = "coral", alpha=0.5, linestyle="dashed", where="mid")
    ax[0].step(t, k_sv[0][1][1], color = "coral", alpha=0.5, linestyle="dashed", where="mid")

    ax[1].step(t, k_sv[1][0], color = "coral", label = "k1", alpha=0.8, where="mid")
    ax[1].step(t, k_sv[1][1][0], color = "coral",  alpha=0.5, linestyle="dashed", where="mid")
    ax[1].step(t, k_sv[1][1][1], color = "coral",  alpha=0.5, linestyle="dashed", where="mid")

    # pb
    ax[0].step(t, pb_sv[0][0], label="pb0", color = "blue", alpha=0.5, where="mid")
    ax[0].step(t, pb_sv[0][1][0], color = "blue", alpha=0.5, linestyle="dashed", where="mid")
    ax[0].step(t, pb_sv[0][1][1], color = "blue", alpha=0.5, linestyle="dashed", where="mid")
    
    ax[1].step(t, pb_sv[1][0], label="pb1", color = "blue", alpha=0.5, where="mid")
    ax[1].step(t, pb_sv[1][1][0], color = "blue", alpha=0.5, linestyle="dashed", where="mid")
    ax[1].step(t, pb_sv[1][1][1], color = "blue", alpha=0.5, linestyle="dashed", where="mid")

    #rb
    ax[0].step(t, r_sv[0][0], color="green", label="r0", alpha=0.5, where="mid")
    ax[0].step(t, r_sv[0][1][0], color="green",  alpha=0.5, linestyle="dashed", where="mid")
    ax[0].step(t, r_sv[0][1][1], color="green",  alpha=0.5, linestyle="dashed", where="mid")
    
    ax[1].step(t, r_sv[1][0], color="green", label="r1", alpha=0.5, where="mid")
    ax[1].step(t, r_sv[1][1][0], color="green",  alpha=0.5, linestyle="dashed", where="mid")
    ax[1].step(t, r_sv[1][1][1], color="green", alpha=0.5, linestyle="dashed", where="mid")
    ax[0].legend()
    ax[1].legend()
    return fig
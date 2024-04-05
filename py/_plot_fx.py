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
    msk = meta[3][1]
    t = meta[3][0]
    print(t)
    print(k_sv[0])
    ax.step(t, sv_true[msk], color="black", label="t0", where="mid")
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

    fig, ax = plt.subplots(1,3,figsize=figsize)

    ttt = meta[1]
    msk = meta[3][1]
    t = meta[3][0]
    print(t)
    print(msk)
    ax[0].step(t, sv_true[0][msk], color="black", label="t0", where="mid")
    ax[1].step(t, sv_true[1][msk], color="black", label="t1", where="mid")
    ax[2].step(t, sv_true[0][msk], color="black", label="t0", where="mid")
    ax[2].step(t, sv_true[1][msk], color="black", label="t1", where="mid", alpha=0.5)
    # kp
    ax[0].step(t, k_sv[0][0], color = "coral", label = "k0", alpha=0.8, where="mid")
    ax[0].step(t, k_sv[0][1][0], color = "coral", alpha=0.5, linestyle="dashed", where="mid")
    ax[0].step(t, k_sv[0][1][1], color = "coral", alpha=0.5, linestyle="dashed", where="mid")

    ax[1].step(t, k_sv[1][0], color = "coral", label = "k1", alpha=0.8, where="mid")
    ax[1].step(t, k_sv[1][1][0], color = "coral",  alpha=0.5, linestyle="dashed", where="mid")
    ax[1].step(t, k_sv[1][1][1], color = "coral",  alpha=0.5, linestyle="dashed", where="mid")

    ax[2].step(t, k_sv[0][0], color = "coral", label = "k0", alpha=0.8, where="mid")
    ax[2].step(t, k_sv[1][0], color = "coral", label = "k1", alpha=0.4, where="mid")

    # pb
    ax[0].step(t, pb_sv[0][0], label="pb0", color = "blue", alpha=0.5, where="mid")
    ax[0].step(t, pb_sv[0][1][0], color = "blue", alpha=0.5, linestyle="dashed", where="mid")
    ax[0].step(t, pb_sv[0][1][1], color = "blue", alpha=0.5, linestyle="dashed", where="mid")
    
    ax[1].step(t, pb_sv[1][0], label="pb1", color = "blue", alpha=0.5, where="mid")
    ax[1].step(t, pb_sv[1][1][0], color = "blue", alpha=0.5, linestyle="dashed", where="mid")
    ax[1].step(t, pb_sv[1][1][1], color = "blue", alpha=0.5, linestyle="dashed", where="mid")

    ax[2].step(t, pb_sv[0][0], label="pb0", color = "blue", alpha=0.7, where="mid")
    ax[2].step(t, pb_sv[1][0], label="pb1", color = "blue", alpha=0.4, where="mid")
    
    #rb
    ax[0].step(t, r_sv[0][0], color="green", label="r0", alpha=0.5, where="mid")
    ax[0].step(t, r_sv[0][1][0], color="green",  alpha=0.5, linestyle="dashed", where="mid")
    ax[0].step(t, r_sv[0][1][1], color="green",  alpha=0.5, linestyle="dashed", where="mid")
    
    ax[1].step(t, r_sv[1][0], color="green", label="r1", alpha=0.5, where="mid")
    ax[1].step(t, r_sv[1][1][0], color="green",  alpha=0.5, linestyle="dashed", where="mid")
    ax[1].step(t, r_sv[1][1][1], color="green", alpha=0.5, linestyle="dashed", where="mid")
    
    ax[2].step(t, r_sv[0][0], color="green", label="r0", alpha=0.7, where="mid")
    ax[2].step(t, r_sv[1][0], color="green", label="r1", alpha=0.4, where="mid")

    ax[0].legend()
    ax[1].legend()
    return fig

def plots3(
    qnt_t,
    # qnt_idx,
    sv_true,
    cph_sv,
    pb_sv,
    r_sv,
    figsize=(16,8)
):

    fig, ax = plt.subplots(1,3,figsize=figsize)
    i = 0
    ax[0].step(qnt_t, sv_true[i,:], color="black", label="true", where="mid",alpha=0.5)
    ax[1].step(qnt_t, sv_true[i,:], color="black", label="true", where="mid",alpha=0.5)
    ax[2].step(qnt_t, sv_true[i,:], color="black", label="true", where="mid",alpha=0.5)
    ax[0].step(qnt_t, cph_sv[i,:], color="red", label="cph", where="mid",alpha=0.5)
    ax[1].step(qnt_t, pb_sv[i,:], color="blue", label="pbart", where="mid",alpha=0.5)
    ax[2].step(qnt_t, r_sv[i,:], color="green", label="rbart", where="mid",alpha=0.5)
    for i in range(1,100):
        ax[0].step(qnt_t, sv_true[i,:], color="black", where="mid",alpha=0.5)
        ax[1].step(qnt_t, sv_true[i,:], color="black", where="mid",alpha=0.5)
        ax[2].step(qnt_t, sv_true[i,:], color="black", where="mid",alpha=0.5)
        ax[0].step(qnt_t, cph_sv[i,:], color="red", where="mid",alpha=0.5)
        ax[1].step(qnt_t, pb_sv[i,:], color="blue", where="mid",alpha=0.5)
        ax[2].step(qnt_t, r_sv[i,:], color="green", where="mid",alpha=0.5)
       

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    return fig

def plots3_3(sv_t_0,cph_sv, pb_sv,r_sv, sv_t_tst_0, cph_sv_tst, pb_sv_tst, qs=1,qe=-1):
    fig, ax = plt.subplots(1,5, figsize = (40,8))
    ax[0].scatter(sv_t_0[:,qs:qe].flatten(), cph_sv[:,qs:qe].flatten(),s=0.1, color = "red", label = "cph")
    ax[0].legend()
    ax[1].scatter(sv_t_0[:,qs:qe].flatten(), pb_sv[:,qs:qe].flatten(),s=0.1, color="green", label="p")
    ax[1].legend()
    ax[2].scatter(sv_t_0[:,qs:qe].flatten(), r_sv[:,qs:qe].flatten(),s=0.1, color = "blue", label = "r")    
    ax[2].legend()
    ax[3].scatter(sv_t_tst_0[:,qs:qe].flatten(), cph_sv_tst[:,qs:qe].flatten(),s=0.1, color = "red", label="cph_tst")
    ax[3].legend()
    ax[4].scatter(sv_t_tst_0[:,qs:qe].flatten(), pb_sv_tst[:,qs:qe].flatten(),s=0.1, color = "green", label = "p_tst")
    ax[4].legend()
    return fig
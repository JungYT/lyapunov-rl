import json
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

with open("./tensorboard_data/N1.json", "r") as N1_json:
    N1 = np.array(json.load(N1_json))
with open("./tensorboard_data/N2.json", "r") as N2_json:
    N2 = np.array(json.load(N2_json))
with open("./tensorboard_data/N3.json", "r") as N3_json:
    N3 = np.array(json.load(N3_json))
with open("./tensorboard_data/Q1.json", "r") as Q1_json:
    Q1 = np.array(json.load(Q1_json))
with open("./tensorboard_data/Q2.json", "r") as Q2_json:
    Q2 = np.array(json.load(Q2_json))
with open("./tensorboard_data/Q3.json", "r") as Q3_json:
    Q3 = np.array(json.load(Q3_json))
with open("./tensorboard_data/EQ1.json", "r") as EQ1_json:
    EQ1 = np.array(json.load(EQ1_json))
with open("./tensorboard_data/EQ2.json", "r") as EQ2_json:
    EQ2 = np.array(json.load(EQ2_json))
with open("./tensorboard_data/EQ3.json", "r") as EQ3_json:
    EQ3 = np.array(json.load(EQ3_json))
with open("./tensorboard_data/L1.json", "r") as L1_json:
    L1 = np.array(json.load(L1_json))
with open("./tensorboard_data/L2.json", "r") as L2_json:
    L2 = np.array(json.load(L2_json))
with open("./tensorboard_data/L3.json", "r") as L3_json:
    L3 = np.array(json.load(L3_json))
with open("./tensorboard_data/L4.json", "r") as L4_json:
    L4 = np.array(json.load(L4_json))
with open("./tensorboard_data/EQL1.json", "r") as EQL1_json:
    EQL1 = np.array(json.load(EQL1_json))
with open("./tensorboard_data/EQL2.json", "r") as EQL2_json:
    EQL2 = np.array(json.load(EQL2_json))

def set_size(width_pt=245.7, fraction=1, subplots=(1,1)):
# def set_size(width_pt=155, fraction=1, subplots=(1,1)):
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27

    # golden_ratio = (5**.5 - 1) / 2
    golden_ratio = 3 / 4

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    # fig_height_in = fig_width_in * golden_ratio
    return (fig_width_in, fig_height_in)
    
tex_fonts = {
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
}

fig, ax = plt.subplots(3, 5, figsize=set_size(width_pt=455, subplots=(3,5)))
fig.suptitle('Return values histories vs. iteration')
fig.supxlabel('Iteration Number')
fig.supylabel('Return value')
fig.subplots_adjust(hspace=.5)
fig.tight_layout()
ax[0, 0].plot(N1[:,1], N1[:,2], 'g')
ax[0, 0].axes.xaxis.set_ticklabels([])
ax[0, 0].axes.yaxis.set_ticklabels([])
ax[0, 0].set_xlabel('(a)')
ax[1, 0].plot(N2[:,1], N2[:,2], 'g')
ax[1, 0].axes.xaxis.set_ticklabels([])
ax[1, 0].axes.yaxis.set_ticklabels([])
ax[1, 0].set_xlabel('(b)')
ax[2, 0].plot(N3[:,1], N3[:,2], 'g')
ax[2, 0].axes.xaxis.set_ticklabels([])
ax[2, 0].axes.yaxis.set_ticklabels([])
ax[2, 0].set_xlabel('(c)')
ax[0, 1].plot(Q1[:,1], Q1[:,2], 'k')
ax[0, 1].axes.xaxis.set_ticklabels([])
ax[0, 1].axes.yaxis.set_ticklabels([])
ax[0, 1].set_xlabel('(d)')
ax[1, 1].plot(Q2[:,1], Q2[:,2], 'k')
ax[1, 1].axes.xaxis.set_ticklabels([])
ax[1, 1].axes.yaxis.set_ticklabels([])
ax[1, 1].set_xlabel('(e)')
ax[2, 1].plot(Q3[:,1], Q3[:,2], 'k')
ax[2, 1].axes.xaxis.set_ticklabels([])
ax[2, 1].axes.yaxis.set_ticklabels([])
ax[2, 1].set_xlabel('(f)')
ax[0, 2].plot(EQ1[:,1], EQ1[:,2], 'm')
ax[0, 2].axes.xaxis.set_ticklabels([])
ax[0, 2].axes.yaxis.set_ticklabels([])
ax[0, 2].set_xlabel('(g)')
ax[1, 2].plot(EQ2[:,1], EQ2[:,2], 'm')
ax[1, 2].axes.xaxis.set_ticklabels([])
ax[1, 2].axes.yaxis.set_ticklabels([])
ax[1, 2].set_xlabel('(h)')
ax[2, 2].plot(EQ3[:,1], EQ3[:,2], 'm')
ax[2, 2].axes.xaxis.set_ticklabels([])
ax[2, 2].axes.yaxis.set_ticklabels([])
ax[2, 2].set_xlabel('(i)')
ax[0, 3].plot(L1[:,1], L1[:,2], 'r')
ax[0, 3].axes.xaxis.set_ticklabels([])
ax[0, 3].axes.yaxis.set_ticklabels([])
ax[0, 3].set_xlabel('(j)')
ax[1, 3].plot(L2[:,1], L2[:,2], 'r')
ax[1, 3].axes.xaxis.set_ticklabels([])
ax[1, 3].axes.yaxis.set_ticklabels([])
ax[1, 3].set_xlabel('(k)')
ax[2, 3].plot(L3[:,1], L3[:,2], 'r')
ax[2, 3].axes.xaxis.set_ticklabels([])
ax[2, 3].axes.yaxis.set_ticklabels([])
ax[2, 3].set_xlabel('(l)')
ax[0, 4].plot(L4[:,1], L4[:,2], 'r')
ax[0, 4].axes.xaxis.set_ticklabels([])
ax[0, 4].axes.yaxis.set_ticklabels([])
ax[0, 4].set_xlabel('(m)')
ax[1, 4].plot(EQL1[:,1], EQL1[:,2], 'b')
ax[1, 4].axes.xaxis.set_ticklabels([])
ax[1, 4].axes.yaxis.set_ticklabels([])
ax[1, 4].set_xlabel('(n)')
ax[2, 4].plot(EQL2[:,1], EQL2[:,2], 'b')
ax[2, 4].axes.xaxis.set_ticklabels([])
ax[2, 4].axes.yaxis.set_ticklabels([])
ax[2, 4].set_xlabel('(o)')
fig.savefig(Path("./ray_results/compare/", "return.pdf"), bbox_inches='tight')



import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif"})
W, H = 25.0, 10.0
fig = plt.figure(figsize=(W / 1.9, H / 1.9), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

def box(x, y, w, h, ec, fc, ls="--", lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25",
                                ec=ec, fc=fc, ls=ls, lw=lw))

def txt(x, y, s, size=11, color="#1b1b1b", weight="normal", ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va)

def arrow(x0, y0, x1, y1, color, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=1.6, ls=ls))

# ---------------- input ----------------
C_IN = "#2e5fa8"
box(0.2, 0.4, 4.6, 9.2, "#7ea3d8", "#eef3fb")
txt(2.5, 9.05, "Input", 15, C_IN, "bold")
txt(2.5, 8.45, "15 features = 3 positions × 5 features", 8.6)
labels = [r"$v_{i-1}$", r"$v_{i}$", r"$v_{i+1}$"]
feats = r"$\log_{10}f,\ \mathrm{Re}(Z),$" + "\n" + r"$\mathrm{Im}(Z),\ |Z|,\ \varphi$"
ys = [6.75, 4.55, 2.35]
for lab, yc in zip(labels, ys):
    box(0.45, yc - 0.95, 4.1, 1.9, C_IN, "white", ls="-", lw=1.2)
    txt(1.05, yc, lab, 14)
    txt(2.75, yc, feats, 9.5)
    for k in range(5):
        ax.add_patch(Circle((4.35, yc + 0.6 - 0.3 * k), 0.09, color=C_IN))
txt(2.5, 0.85, r"$x_i=[v_{i-1},\,v_i,\,v_{i+1}]$" + "\n(neighbouring frequencies, one sweep)", 8)

# ---------------- hidden layers ----------------
layers = [(6.0, "Hidden layer 1", "128 units", "tanh + Dropout (0.2)", "#1f9fc9", "#eaf6fb"),
          (8.9, "Hidden layer 2", "128 units", "tanh + Dropout (0.2)", "#14998a", "#e8f6f3"),
          (11.8, "Hidden layer 3", "64 units", "tanh", "#2f9e44", "#ebf6ec")]
node_y = [7.3, 6.0, 4.7, 2.4]
prev = [(4.44, yc + 0.6 - 0.3 * k) for yc in ys for k in range(5)]
for x0, name, units, act, col, fc in layers:
    box(x0, 0.4, 2.5, 9.2, col, fc)
    txt(x0 + 1.25, 9.05, name, 10.2, col, "bold")
    txt(x0 + 1.25, 8.45, units, 10.5)
    cx = x0 + 1.25
    pts = [(cx, y) for y in node_y]
    for (px, py) in prev:
        for (qx, qy) in pts:
            ax.plot([px, qx], [py, qy], color=col, lw=0.35, alpha=0.35, zorder=1)
    for (qx, qy) in pts:
        ax.add_patch(Circle((qx, qy), 0.42, color=col, zorder=3))
    for dy in (3.85, 3.55, 3.25):
        ax.add_patch(Circle((cx, dy), 0.06, color="#333333", zorder=3))
    txt(cx, 1.0, act, 8.4)
    prev = pts

# ---------------- output ----------------
C_OUT = "#e07b2a"
box(14.7, 2.0, 2.6, 6.0, C_OUT, "#fdf1e6")
txt(16.0, 7.45, "Output", 13.5, C_OUT, "bold")
txt(16.0, 6.85, "2 units, softmax", 10)
for (qx, qy) in prev:
    for oy in (5.4, 3.9):
        ax.plot([qx, 16.0], [qy, oy], color=C_OUT, lw=0.4, alpha=0.4, zorder=1)
ax.add_patch(Circle((15.75, 5.4), 0.48, color=C_OUT, zorder=3)); txt(16.75, 5.4, r"$p_0$", 13)
ax.add_patch(Circle((15.75, 3.9), 0.48, color=C_OUT, zorder=3)); txt(16.75, 3.9, r"$p_1$", 13)
txt(16.0, 2.65, r"$p_1=P(\mathrm{SoC}\geq 60\%\mid x)$", 8)

# ---------------- losses ----------------
C_PH, C_CE, C_TOT = "#c0561b", "#2f8a3e", "#6a3fb5"
box(18.0, 5.35, 4.7, 4.25, C_PH, "#fdf0e8", ls="-")
txt(20.35, 9.05, "Physics-guided term", 12, C_PH, "bold")
txt(20.35, 7.95, r"$\mathcal{L}_{\mathrm{phys}}=\mathbb{E}_x\!\left[p_1\,z_{\mathrm{norm}}\,\mathbb{1}\{z_{\mathrm{norm}}>0.5\}\right]$", 9.0)
txt(20.35, 6.55, r"$z_{\mathrm{norm}}$: centre-point $|Z|$, min–max scaled" + "\non the training set; penalises a high\nHigh-SoC probability on high-impedance\nwindows (Randles-motivated prior)", 8.2)
box(18.0, 0.4, 4.7, 4.25, C_CE, "#ebf6ec", ls="-")
txt(20.35, 4.1, "Cross-entropy term", 12, C_CE, "bold")
txt(20.35, 2.95, r"$\mathcal{L}_{\mathrm{CE}}=-\sum_{c\in\{0,1\}} y_c\log p_c$", 11)
txt(20.35, 1.5, "class labels: 0 = Low-Mid SoC (<60%)\n1 = High SoC (≥60%)", 8.5)
arrow(17.3, 5.6, 17.95, 7.2, C_PH, "--")
arrow(17.3, 3.6, 17.95, 2.6, C_CE, "--")
txt(17.62, 6.85, r"$p_1$", 11, C_PH)
txt(17.62, 2.35, r"$p_0,p_1$", 9.5, C_CE)

box(23.05, 2.9, 1.85, 4.2, C_TOT, "#f3eefb", ls="-")
txt(23.92, 6.45, "Total loss", 11.5, C_TOT, "bold")
txt(23.92, 5.25, r"$\mathcal{L}_{\mathrm{total}}=$", 11)
txt(23.92, 4.45, r"$\mathcal{L}_{\mathrm{CE}}+\lambda\,\mathcal{L}_{\mathrm{phys}}$", 10)
txt(23.92, 3.55, r"($\lambda=1.0$)", 10, C_TOT)
arrow(22.7, 7.5, 23.4, 7.15, C_TOT)
arrow(22.7, 2.5, 23.4, 2.85, C_TOT)

fig.savefig("figure3_architecture.png", dpi=300, facecolor="white")
print("ok")
